# 15 — Training on Kaggle (2× T4 + cached dataset)

Running a full `1024/320` training pass on Kaggle is fastest when you split it
into two notebooks:

1. **Build the tokenised cache once on a CPU notebook** (free, no GPU quota
   spent) and save the output folder as a **Kaggle Dataset**.
2. **Train on a 2× T4 GPU notebook** with that dataset attached — startup drops
   from several minutes to **~1–2 seconds** because HuggingFace download,
   parsing, and tokenisation are skipped completely.

---

## 1. One-time notebook setup cell

Run this first in any fresh Kaggle notebook (both CPU and GPU):

```python
import os

!curl -LsSf https://astral.sh/uv/install.sh | sh
os.environ["PATH"] = f"/root/.local/bin:{os.environ['PATH']}"
os.environ["UV_PROJECT_ENVIRONMENT"] = "/tmp/systemone-venv"

if not os.path.exists("/kaggle/working/systemone"):
    !git clone https://github.com/AvikArefin/systemone.git /kaggle/working/systemone

os.chdir("/kaggle/working/systemone")
!git pull
!uv sync
```

> Keeping `UV_PROJECT_ENVIRONMENT` in `/tmp/systemone-venv` avoids filling
> `/kaggle/working` (which counts against Kaggle's output quota).

*(Optional)* If you added a free read-only HuggingFace token under **Add-ons →
Secrets** as `HF_TOKEN`, load it before downloading to avoid anonymous Hub rate
limits:

```python
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

---

## 2. Build the cache once (CPU-only notebook)

Turn **Accelerator = None** (CPU) and **Internet = On**, run Setup above, then:

```python
import os

os.chdir("/kaggle/working/systemone")
!git pull

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["UV_PROJECT_ENVIRONMENT"] = "/tmp/systemone-venv"
os.environ["SYSTEMONE_CHECKPOINT_DIR"] = "/kaggle/working/artifacts"
os.environ["SYSTEMONE_NAME"] = "bekko_1024"
os.environ["SYSTEMONE_STATE_TOKENS"] = "1024"
os.environ["SYSTEMONE_QUERY_TOKENS"] = "320"
os.environ["BEKKO_CASES"] = "1000"
os.environ["SYSTEMONE_CACHE_ONLY"] = "1"
# Optional: drop individual rows that still overflow 1024/320
# os.environ["SYSTEMONE_FIT_ONLY"] = "1"

!uv run python scripts/train_main.py
```

This downloads all needed parquet files in parallel (`snapshot_download`),
trains a fresh 3,072-piece tokenizer, encodes the flat `int32` caches, and exits
without training. You should see:

```text
/kaggle/working/artifacts/bekko_1024/
  tokenizer.model
  tokenizer.vocab
  bekko_train_cache.pt        (or bekko_train_cache_fit.pt)
  test_cache.pt               (or test_cache_fit.pt)
  probe_cache.pt              (only if len(train) < len(test))
```

### Turn the output into a Kaggle Dataset (zero local upload)

1. Click **Save Version → Save & Run All** (or open the **Output** panel on the
   right after the cell finishes).
2. On `/kaggle/working/artifacts/bekko_1024`, click **⋮ → New Dataset** and name
   it e.g. **`bekko-1024-cache`**.

Kaggle creates the dataset server-side in a few seconds.

---

## 3. Train on 2× T4 GPUs (~1–2s startup)

Create a GPU notebook with **Accelerator = GPU T4 x2** and **Internet = On**
(only needed for `git pull` / `uv sync`).

1. Click **+ Add Input → Datasets → Your Datasets** and attach
   **`bekko-1024-cache`**. It mounts read-only at
   `/kaggle/input/bekko-1024-cache/`.
2. Run the Setup cell from Section 1.
3. Run the training cell:

```python
import os
import pathlib

os.chdir("/kaggle/working/systemone")
!git pull

# Symlink read-only cached tensors + tokenizer into writable checkpoint dir
cache_src = pathlib.Path("/kaggle/input/bekko-1024-cache")
dst = pathlib.Path("/kaggle/working/artifacts/bekko_1024")
dst.mkdir(parents=True, exist_ok=True)
for f in cache_src.glob("*"):
    link = dst / f.name
    if not link.exists():
        link.symlink_to(f)

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["UV_PROJECT_ENVIRONMENT"] = "/tmp/systemone-venv"
os.environ["SYSTEMONE_CHECKPOINT_DIR"] = "/kaggle/working/artifacts"
os.environ["SYSTEMONE_NAME"] = "bekko_1024"
os.environ["SYSTEMONE_STATE_TOKENS"] = "1024"
os.environ["SYSTEMONE_QUERY_TOKENS"] = "320"
os.environ["SYSTEMONE_CHECKPOINT"] = "1"
os.environ["SYSTEMONE_MAX_ROWS"] = "64"
os.environ["SYSTEMONE_GRAD_ACCUM"] = "4"
os.environ["SYSTEMONE_COMPILE"] = "1"
os.environ["BEKKO_CASES"] = "1000"
os.environ["SYSTEMONE_EPOCHS"] = "2"
# Must match what you built the cache with:
# os.environ["SYSTEMONE_FIT_ONLY"] = "1"

!uv run torchrun --standalone --nproc_per_node=2 scripts/train_main.py
```

On startup you should see:

```text
cache hit in 0.85s vocab=3072 train=... test=... probe=... (skipped HuggingFace)
```

- **Effective batch size:** `64 rows × 4 grad_accum × 2 GPUs = 512 rows/step`.
- **Precision:** `train_main.py` auto-enables AMP on CUDA and picks `fp16` +
  `GradScaler` on T4 (which lacks native `bf16`).
- **Outputs:** saved to `/kaggle/working/artifacts/bekko_1024/model.pt` and
  `training_state.pt`.

---

## 4. Resume training for more epochs

Because `training_state.pt` is saved at the end of every epoch (including
optimiser and LR schedule), you can continue a run by setting:

```python
os.environ["SYSTEMONE_RESUME"] = "1"
os.environ["SYSTEMONE_EPOCHS"] = "6"   # target total epochs, not additional
```

*(If resuming in a brand-new Kaggle session, attach your previous notebook's
output and copy `training_state.pt` into `/kaggle/working/artifacts/bekko_1024/`
so it can be overwritten as new epochs finish.)*

---

## 5. Evaluate the trained checkpoint

After training finishes, compute the baseline-adjusted **Task Avg** on the full
test split:

```python
os.environ["SYSTEMONE_CHECKPOINT_DIR"] = "/kaggle/working/artifacts"
os.environ["SYSTEMONE_EVAL_CHECKPOINT"] = "bekko_1024"
!uv run python scripts/eval.py
```

---

## Tuning & troubleshooting

| Symptom / goal | Setting |
|---|---|
| OOM on longest `1024/320` batches (15 GB T4) | Lower `SYSTEMONE_MAX_ROWS=32` and raise `SYSTEMONE_GRAD_ACCUM=8` (keeps 512 effective rows/step). |
| Plenty of free VRAM on T4 | Bump `SYSTEMONE_MAX_ROWS=128` and `SYSTEMONE_GRAD_ACCUM=2`. |
| Slow startup or recompilations with `torch.compile` + checkpointing | Set `SYSTEMONE_COMPILE=0`. |
| Cache not hitting (re-downloading from HF) | Verify `SYSTEMONE_STATE_TOKENS`, `SYSTEMONE_QUERY_TOKENS`, and `SYSTEMONE_FIT_ONLY` match the values used when building the cache; `EncodedSet.compatible()` rejects mismatched budgets/tokenizers. |
| Want a fast sanity check first | Add `SYSTEMONE_DEV=1` (5 configs, 200 cases/config, saves to `<name>_dev`). |
