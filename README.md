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
```

Environment knobs: `SYSTEMONE_EPOCHS`, `SYSTEMONE_BATCH`, `BEKKO_CASES`,
`SYSTEMONE_CHECKPOINT_DIR` (defaults to `artifacts/`).

## Data

- **Evaluation:** S1MB (`hotchpotch/s1mb-dataset`, 106 subsets, 26k judgments).
  Test-only; scores on the public leaderboard are baseline-adjusted, not accuracy.
- **Training:** `hotchpotch/bekko-system-one-dataset-v0` — 8.4M judgments in the
  same schema, per-source licensed. Train on this; evaluate on S1MB test.

Loading is capped per subset (`BEKKO_CASES`) because a full pass is ~262k steps.

## Measured

Raw accuracy on the S1MB test split, 1M parameters, 6-candidate cap:

| Training data | Examples | `noul` | `choice` | `score` | overall |
|---|---:|---:|---:|---:|---:|
| 20 S1MB configs (leaks test rows) | 2,446 | 0.659 | 0.222 | 0.176 | 0.417 |
| all 106 S1MB configs (leaks test rows) | 26,264 | 0.721 | 0.323 | 0.508 | 0.551 |
| bekko corpus, 300/subset | 89,442 | 0.701 | 0.335 | 0.510 | 0.541 |
| bekko corpus, 1000/subset | 284,877 | 0.698 | 0.308 | 0.526 | 0.534 |

The first two rows trained on the benchmark's own test rows, so they are not
comparable to published results. Scaling corpus volume past ~90k examples did
not help: the 1M model is capacity-bound, not data-bound.

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
