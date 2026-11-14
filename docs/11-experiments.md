# 11 — Experiments and results

Every number here was produced on this machine (Apple MPS, 1M-parameter model)
unless stated otherwise. Commands are in each section.

## Headline table

Raw accuracy on the S1MB test split, per decision type:

| Training data | Examples | noul | choice | score | overall |
|---|---:|---:|---:|---:|---:|
| 20 S1MB configs, 6 epochs *(leaks test rows)* | 2,446 | 0.659 | 0.222 | 0.176 | 0.417 |
| all 106 S1MB configs, 10 epochs *(leaks test rows)* | 26,264 | 0.721 | 0.323 | 0.508 | 0.551 |
| bekko corpus 300/subset, 1 epoch | 89,442 | 0.701 | 0.335 | 0.510 | **0.541** |
| bekko corpus 1000/subset, 2 epochs | 284,877 | 0.698 | 0.308 | 0.526 | 0.534 |

Chance: `choice` ≈ 0.125 with 8 options, `score` ≈ 0.25 with 4 levels, `noul`
0.5. Read [Metrics](10-metrics.md) before comparing the last two rows to any
published score — ours are raw accuracy, published ones are baseline-adjusted.

## What each change actually bought

### More data (the big one)

`0.417 → 0.551`. Going from 20 configs to all 106 took `score` from *below
chance* to 0.508. Volume, not architecture, was the blocker.

### The flat token cache

Same 20-config, 6-epoch run, before vs after replacing Python list-of-lists with
`int32` buffers + offsets:

```
before  394.3 s
after   295.8 s     (1.33x, ~98 s saved)
```

Collate alone measured 1.78 ms → 0.53 ms per batch of 32. An earlier claim that
"tokenisation is 80% of the step" was **wrong** — tokenisation was ~1.8 ms; the
real cost was the per-row `torch.tensor(seq)` loop, which the flat layout removes.

The cache also dropped storage to ~1 MB for 26k examples (no padding stored) and
made reloads 5 ms.

### SDPA attention instead of hand-written

Swapping the manual attention (repeat KV, matmul, softmax) for
`F.scaled_dot_product_attention` with `enable_gqa=True`:

```
attention layer   : 2.52x faster on CPU, 3.30x on MPS
end-to-end step   : 247.1 ms → 216.3 ms   (1.14x)
numerical check   : max abs diff 1.3e-08  (checkpoint still loads)
```

On a real run the end-to-end gain was only ~3%, because at that point model
compute was not the bottleneck.

### Native RMSNorm

Replaced a hand-written RMSNorm with `torch.nn.RMSNorm`. Verified
`max_diff = 0.0` and the same `*.weight` parameter names, so existing checkpoints
load unchanged.

### Seeding

`scripts/train_s1mb.py` originally did not seed at all, so every run initialised
differently. Now every script calls `set_seed(seed)` and `build_model(cfg, seed)`,
and the seed is stored in `model.pt`.

## Negative and surprising results

**Scaling the corpus stopped helping.** 89k → 285k examples, 1 → 2 epochs:
0.541 → 0.534, and eval loss rose from 1.098 to 1.106 between epochs. Train
accuracy on `choice` was still only 0.418. Conclusion: the 1M model is
**capacity-bound**, not data-bound.

**A `score` head below chance.** Early runs scored 0.176 on `score` against a
0.25 baseline, with only 195 training examples. Adding data fixed it (0.51). Not
a modelling bug — a starvation bug.

**Calibration is not accuracy.** Soft labels plus temperature scaling improved
ECE from 0.117 to 0.069 in a run where test accuracy actually *fell* (0.401),
because dropping a dataset cut `noul` training volume by 70%. Confidence and
knowledge are separate axes.

## Reproduce

```bash
# clean corpus run, ~20 min on MPS
BEKKO_CASES=300 SYSTEMONE_EPOCHS=1 uv run python scripts/train_bekko.py

# larger run, ~2 h
BEKKO_CASES=1000 SYSTEMONE_EPOCHS=2 SYSTEMONE_BATCH=64 uv run python scripts/train_bekko.py

# S1MB-config run (trains on the benchmark — comparison only)
SYSTEMONE_EPOCHS=10 uv run python scripts/train_s1mb.py

# synthetic sanity task, seconds
uv run python scripts/train_demo.py
```

Checkpoint goes to `artifacts/<name>/`; override with
`SYSTEMONE_CHECKPOINT_DIR`.

## What the results imply for next steps

1. **Measure properly first.** A Task Avg number is the only thing that can be
   compared to the published table.
2. **Capacity, not data.** 1M → 4.2M is the supported next move; the 17M/68M/400M
   bekko scaling (27.57 / 40.46 / 50.60 Task Avg) suggests real headroom.
3. **Distillation** is the only lever that adds information rather than
   reweighting what we have. Laya is Apache-2.0 and runnable.

Next: [Code map →](12-code-map.md)
