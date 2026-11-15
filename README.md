# systemone

`TinyJev` — a small System One decision model. One forward pass answers typed
questions about a shared state: `noul` (yes/no), `choice` (pick an option), and
`score` (numeric rubric). No text is generated, so output errors are structural
rather than syntactic.

Target deployment is a microcontroller: the 1M-parameter model fits the 8 MB
PSRAM budget in int8/ternary, and `d_model=128, n_layers=4` is the 4.2M ceiling.

## Documentation

A full guide — architecture, data pipeline, training, inference, metrics,
experiments, and a file-by-file code map — lives in
[`docs/`](docs/README.md). It assumes no ML background.

## Layout

```
src/systemone/
  model.py      TinyJev: prefix-LM over state + candidates, SDPA with GQA
  data.py       Example/Batch, EncodedSet token cache, soft-label loss
  train.py      train/evaluate, warmup + cosine schedule
  calibrate.py  temperature scaling, ECE, per-kind accuracy
  s1mb.py       loaders for S1MB evaluation and the bekko training corpus
  inference.py  answer_request: one request in, typed answers out
  checkpoint.py save/load under artifacts/<name>/
scripts/
  train_bekko.py  train on the bekko typed-decision corpus
  train_s1mb.py   train/evaluate on S1MB configs
  train_demo.py   small synthetic task, end-to-end
  smoke.py        construction/inference sanity check
```

## Install and run

```bash
uv sync
uv run python scripts/smoke.py
uv run python scripts/train_bekko.py          # BEKKO_CASES=1000, epochs=2
uv run python scripts/train_demo.py
uv run python scripts/eval_s1mb.py            # baseline-adjusted Task Avg
```

Environment knobs: `SYSTEMONE_EPOCHS`, `SYSTEMONE_BATCH`, `BEKKO_CASES`,
`SYSTEMONE_CHECKPOINT_DIR` (defaults to `artifacts/`).

## Tests

```bash
uv run pytest                     # fast suite, ~3 s
uv run pytest -m slow             # dataset-backed, needs the HF cache
```

The fast suite covers encoding, the token cache, model invariants, scoring maths,
and schema validation. The slow suite asserts on the real S1MB corpus shapes
(151-option choices, 600-token instructions, 40k-token states) and is skipped when
the HuggingFace cache is absent.

## Data

- **Evaluation:** S1MB (`hotchpotch/s1mb-dataset`, 106 subsets, 26k judgments).
  Test-only; scores on the public leaderboard are baseline-adjusted, not accuracy.
- **Training:** `hotchpotch/bekko-system-one-dataset-v0` — 8.4M judgments in the
  same schema, per-source licensed. Train on this; evaluate on S1MB test.

Loading is capped per subset (`BEKKO_CASES`) because a full pass is ~262k steps.

## Measured

The first clean run, after the encoding fix. Raw accuracy on the S1MB test split,
1M parameters:

| | `noul` | `choice` | `score` | overall |
|---|---:|---:|---:|---:|
| before the fix | 0.698 | 0.308 | 0.526 | 0.534 |
| after the fix (1 epoch, 300/subset) | 0.677 | **0.362** | 0.511 | 0.539 |

`choice` is the kind the bug damaged, and the only one that improved.

Raw accuracy is not the leaderboard metric. The comparable number:

```
Task Avg = 1.04     (laya 13.36, bekko-17m 27.57, jev 59.59)
```

That is low, and it is the honest starting point. `score` is flat zero across all
21 of its benchmarks. See `docs/10-metrics.md` and run
`scripts/eval_s1mb.py` to reproduce.

## Artifacts

`artifacts/<name>/` holds `model.pt`, `tokenizer.model`, `tokenizer.json`, and
optional token caches (`*_cache.pt`). These are gitignored. Caches validate
themselves against a tokenizer fingerprint and rebuild automatically when it
changes.

## Design notes

- Tokens are cached as flat `int32` buffers plus offsets (no padded storage);
  batches pad only their own rows. Rebuilding is cheap once HF files are cached.
- Attention uses `F.scaled_dot_product_attention` with an arbitrary prefix mask
  and native GQA; numerics match the earlier hand-written path.
- `torch.nn.RMSNorm` is used directly.
- Seeding covers Python, NumPy, and torch. MPS kernels are not bit-reproducible;
  force CPU for byte-exact reruns.
