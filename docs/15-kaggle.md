# 15 — Training on Kaggle (2× T4)

Run everything in **one Kaggle notebook** with **Persistence → Files only**:

1. Build the tokenised cache once on **CPU** (saves GPU quota) into
   `/kaggle/working/artifacts/bekko_1024/`.
2. Switch the same notebook's accelerator to **GPU T4 x2** — persistence keeps
   `/kaggle/working/artifacts/bekko_1024/` intact across the restart.
3. Run Setup + Train — instant cache hit (~1–2s), zero HuggingFace downloads.

---

## Notebook settings

- **Internet:** On
- **Persistence:** Files only (keeps `/kaggle/working/systemone` and
  `/kaggle/working/artifacts` across restarts and accelerator switches)
- **Dependency Manager:** Off (this repo requires Python ≥ 3.14 via `uv`, which
  conflicts with Kaggle's base Python image)
- **Accelerator:** **None** for Cell 2 (cache build), then **GPU T4 x2** for
  Cells 3–5

---

## Cell 1 — Setup (run after every restart)

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

---

## Cell 2 — Build cache once (Accelerator: None / CPU)

```python
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["SYSTEMONE_CHECKPOINT_DIR"] = "/kaggle/working/artifacts"
os.environ["SYSTEMONE_NAME"] = "bekko_1024"
os.environ["SYSTEMONE_STATE_TOKENS"] = "1024"
os.environ["SYSTEMONE_QUERY_TOKENS"] = "320"
os.environ["BEKKO_CASES"] = "1000"

!uv run python scripts/build_cache.py
```

Once this finishes, switch **Accelerator → GPU T4 x2** and re-run **Cell 1**.

---

## Cell 3 — Train (Accelerator: GPU T4 x2)

```python
os.environ["PYTHONUNBUFFERED"] = "1"
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

!uv run torchrun --standalone --nproc_per_node=2 scripts/train_main.py
```

Look for `cache hit in ... (skipped HuggingFace)` at startup. Outputs are
written to `/kaggle/working/artifacts/bekko_1024/model.pt` and
`training_state.pt`.

---

## Cell 4 — Resume for more epochs (optional)

```python
os.environ["SYSTEMONE_RESUME"] = "1"
os.environ["SYSTEMONE_EPOCHS"] = "6"

!uv run torchrun --standalone --nproc_per_node=2 scripts/train_main.py
```

---

## Cell 5 — Evaluate

```python
os.environ["SYSTEMONE_CHECKPOINT_DIR"] = "/kaggle/working/artifacts"
os.environ["SYSTEMONE_EVAL_CHECKPOINT"] = "bekko_1024"

!uv run python scripts/eval.py
```

---

## Options and tuning

| Option | When to use it |
|---|---|
| `HF_TOKEN` | Load a free read-only token from **Add-ons → Secrets** (`UserSecretsClient().get_secret("HF_TOKEN")`) before Cell 2 to avoid anonymous HuggingFace rate limits. |
| `SYSTEMONE_MAX_ROWS=32` + `SYSTEMONE_GRAD_ACCUM=8` | If 15 GB T4 hits OOM on the longest `1024/320` batches (keeps 512 effective rows/step). |
| `SYSTEMONE_MAX_ROWS=128` + `SYSTEMONE_GRAD_ACCUM=2` | If T4 has plenty of free VRAM and you want higher throughput. |
| `SYSTEMONE_COMPILE=0` | If `torch.compile` + gradient checkpointing triggers slow recompiles on T4. |
| `SYSTEMONE_DEV=1` | Fast pipeline sanity check (5 configs, 200 cases/config, saves to `<name>_dev`). |
| `SYSTEMONE_FIT_ONLY=1` | Diagnostic only: strictly drops every row (in both train and test) exceeding the budget by ≥1 token. Leave unset for normal runs (`UNFIT_CONFIGS` is already excluded by default). |

### Notes

- **Reusing the cache in another notebook:** In Kaggle's Output panel, click
  **⋮ → New Dataset** on `/kaggle/working/artifacts/bekko_1024`, attach it to
  the new notebook, and symlink `/kaggle/input/<dataset>/*` into
  `/kaggle/working/artifacts/bekko_1024/`.
- **Why `/kaggle/working/artifacts` instead of `systemone/artifacts`:** Keeps
  checkpoints and caches safe if you delete and re-clone
  `/kaggle/working/systemone`, and puts outputs at the top level of
  `/kaggle/working`.
- **Why `UV_PROJECT_ENVIRONMENT=/tmp/systemone-venv`:** Avoids filling
  `/kaggle/working` with ~6 GB / 40k+ venv files (which slows down **Save
  Version** and eats the 20 GB output quota).
- **Cache mismatch:** `EncodedSet.compatible()` checks budget and tokenizer
  fingerprint; if `SYSTEMONE_STATE_TOKENS`, `SYSTEMONE_QUERY_TOKENS`, or
  `SYSTEMONE_FIT_ONLY` differ between Cell 2 and Cell 3, it will fall back to
  downloading from HuggingFace.
