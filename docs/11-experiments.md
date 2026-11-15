# 11 — Experiments and results

Every number here was produced on this machine (Apple MPS, 1M-parameter model)
unless stated otherwise. Commands are in each section.

> **Read this first.** Every measurement below was taken with a bug that deleted
> the candidate text from a large fraction of rows. The numbers are kept because
> they are what we actually ran, but the *conclusions drawn from them were wrong*
> and are corrected inline. See [What we got wrong](#what-we-got-wrong) before
> trusting any row of the table.

## Headline table

Raw accuracy on the S1MB test split, per decision type. **These runs predate the
truncation fix**, so they measure a model that often could not see its options.

| Training data | Examples | noul | choice | score | overall |
|---|---:|---:|---:|---:|---:|
| 20 S1MB configs, 6 epochs *(leaks test rows)* | 2,446 | 0.659 | 0.222 | 0.176 | 0.417 |
| all 106 S1MB configs, 10 epochs *(leaks test rows)* | 26,264 | 0.721 | 0.323 | 0.508 | 0.551 |
| bekko corpus 300/subset, 1 epoch | 89,442 | 0.701 | 0.335 | 0.510 | **0.541** |
| bekko corpus 1000/subset, 2 epochs | 284,877 | 0.698 | 0.308 | 0.526 | 0.534 |

**Do not read these as skill.** Measured on the S1MB test split, the trivial
policies already score:

| kind | best trivial policy | accuracy |
|---|---|---:|
| `noul` | always answer false (71.5% of labels) | 0.715 |
| `choice` | always pick position 0 | 0.254 |
| `score` | always pick the modal level | 0.501 |

So the 0.698 on `noul` is *worse than a constant*, and the 0.526 on `score` is 2.5
points above a constant. An earlier version of this page called `choice` chance
≈ 0.125 "with 8 options", which understated the baseline by half. Published scores
are baseline-adjusted; ours are raw. See [Metrics](10-metrics.md).

## What we got wrong

### The model could not see its candidate

Rows were truncated front-only to 48 tokens. A row is `instructions ␟ candidate`,
and instructions run to 600+ tokens on S1MB, so the candidate was deleted. Measured
on the eval cache that shipped:

| | eval | train |
|---|---:|---:|
| candidate rows with no separator at all (entire candidate gone) | **44.8%** | 34.4% |
| `choice` groups with duplicate identical rows | 28.3% | 5.4% |
| `choice` groups collapsed to one encoding | **27.1%** | 5.4% |
| `choice` accuracy ceiling imposed by the encoding | **0.808** | — |

Of the 0.308 `choice` accuracy, 1,679 groups (19.2%) were wrong *because the gold
row was byte-identical to the row the model picked* — provably unlearnable. Fixing
the encoding alone should move `choice` from ~0.31 to ~0.50 with the same weights.

### "Capacity-bound, not data-bound" was unsupported

That conclusion was drawn from 89k → 285k examples not helping. But the data it
scaled was corrupted in the same way at both sizes, so the experiment could not
distinguish "more data does not help" from "more corrupted data does not help".

The claim is **withdrawn, not refuted.** Capacity is still plausible — 1M sits far
below bekko-17m's 27.57 Task Avg — but it has not been tested on clean data yet.

### "Volume, not architecture, was the blocker"

Also withdrawn. The 0.417 → 0.551 gain came from adding configs, but the rows being
added had candidates. Architecture *was* a blocker, and it was never varied.

### The `score` baseline was wrong

Early runs scoring 0.176 were called "below chance" against an assumed 0.25 on 4
levels. Two things were wrong with that: 5-level scales dominate (4,686 of 5,148
decisions), and the biased modal level means the best constant is **0.501**, not
0.25. So 0.176 was worse than the doc thought, not merely "below chance". The fix
— feeding it more data — was right; the severity was understated by 2×.

## What each change actually bought

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
load unchanged. (The parameter-count formula was updated alongside it and was
wrong; it is now measured rather than computed. See
[Architecture](03-architecture.md).)

### Seeding

`scripts/train_s1mb.py` originally did not seed at all, so every run initialised
differently. Now every script calls `set_seed(seed)` and `build_model(cfg, seed)`,
and the seed is stored in `model.pt`.

## Negative and surprising results that still stand

**Calibration is not accuracy.** Soft labels plus temperature scaling improved
ECE from 0.117 to 0.069 in a run where test accuracy actually *fell* (0.401),
because dropping a dataset cut `noul` training volume by 70%. Confidence and
knowledge are separate axes.

**Compute was not the bottleneck.** SDPA made attention 3.3× faster and the step
only 3% faster. Both are true; the second is the one that matters.

## Where the existing checkpoint actually scores

Scoring `artifacts/bekko_tiny` with the new adjusted metric
(`scripts/eval_s1mb.py`), 12 subsets:

```
choice: raw 0.28 vs best trivial 0.29    -> adjusted 0.00
noul:   balanced accuracy 0.570          -> adjusted 0.14
score:  MAE worse than the median        -> adjusted < 0
Task Avg 0.45
```

That is not a verdict on the architecture — the weights were trained on the
corrupted encoding. It is a verdict on the old numbers: a model reported at
"0.534 overall" is at roughly zero adjusted skill. Retrain before drawing any
conclusion from it.

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

# baseline-adjusted Task Avg on S1MB (comparable to the published table)
SYSTEMONE_EVAL_CHECKPOINT=bekko_tiny uv run python scripts/eval_s1mb.py
```

Checkpoint goes to `artifacts/<name>/`; override with
`SYSTEMONE_CHECKPOINT_DIR`.

## What the results imply for next steps

1. **Retrain on the fixed encoding.** Every number above is from rows that lost
   their candidates. Nothing else can be interpreted until this is redone.
2. **Then measure with the adjusted metric.** `scripts/eval_s1mb.py` now produces
   a Task Avg over 137 benchmarks with no silent drops, so the capacity question
   becomes answerable.
3. **Then test capacity.** 1M → 4.2M is the supported next move; the
   17M/68M/400M bekko scaling (27.57 / 40.46 / 50.60 Task Avg) suggests headroom,
   but "the 1M model is capacity-bound" is a hypothesis, not a finding.
4. **Distillation** is the only lever that adds information rather than
   reweighting what we have. Laya is Apache-2.0 and runnable.

Next: [Code map →](12-code-map.md)
