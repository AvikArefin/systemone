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

## Raising the budget at inference makes it worse

With the mask fixed it became safe to ask the original question: does the
checkpoint score better when it can see more? Evaluated on the full S1MB test
split (106 configs, 14,009 cases, 26,269 decisions), `scripts/measure_state_budget.py`:

```
state/query   noul  choice  score  Task Avg | on the 12,520 rows that 192 cut
   192/64     2.71    0.40   0.00    1.04    |   3.44   0.81   0.04   1.43
   192/128    2.34    0.21   0.00    0.85    |   3.13   0.46   0.04   1.21
   768/64     1.30    0.12   0.00    0.47    |   1.65   0.28   0.08   0.67
   768/128    2.09    0.11   0.00    0.73    |   2.61   0.27   0.08   0.99
```

**More context made every kind worse.** The right-hand block isolates rows whose
state was actually being cut, so it is the fair comparison, and it drops from
1.43 to 0.67.

The reason is not capacity: the checkpoint was trained with `max_state_tokens=192`,
so anything beyond 192 is two distribution shifts at once — positions the RoPE
never saw, and a different head/tail cut of the same text. The 72.7% of state
tokens that never reach the model are a *training* limitation, not an inference
one. Bumping the budget is only worth doing together with a retrain at the new
budget, and until then 192 is the right serving value.

`192/64` reproduces the clean run's Task Avg of **1.04** exactly, which is also
the end-to-end check that the attention rewrite changed no numbers.

## The attention mask was the thing that crashed the machine

Raising the state budget from 192 to 768 rebooted the machine. Twice. The kernel
log names it:

```
panicString: SOCD report detected: (iBoot async abort)
```

That is the GPU driver aborting, not the OS killing a process. Two mistakes
multiplied:

1. **A batch was 16 examples, not 16 rows.** Each example expands into one row per
   candidate, and an S1MB `choice` can carry thousands — the largest measured is
   2,416. So "16 examples" was up to 2,416 rows.
2. **The attention mask was a full grid** of shape `[rows, tokens, tokens]`, built
   per forward. At 576 rows and 791 tokens that is 360 MB of booleans before the
   kernel touches it.

Memory per forward at the two configurations, measured with
`scripts/measure_attention_memory.py` (attention call only, 4 heads, 32-dim):

```
rows   full grid mask   split mask   full grid   split
  16            13 MB         2 MB       14 ms     7 ms
  64            51 MB         7 MB       18 ms    15 ms
 128           103 MB        15 MB       35 ms    28 ms
 256           206 MB        30 MB       89 ms    52 ms
```

Two fixes, both kept:

- **Batch by rows, not examples.** `EncodedSet.batches(max_rows)` groups examples
  until the row count reaches the cap, so a single 2,416-candidate example forms
  its own batch instead of dragging fifteen others into a 2,400-row grid.
  `batch_size` is gone from `train`/`evaluate`/`collect_logits`/`probabilities_by_group`;
  everything takes `max_rows` (default 256).
- **No grid.** Each block now makes two attention calls: the state half uses a
  per-key padding flag, the query half uses a causal rule that costs
  `query_len × total` per row instead of `total²`. See
  [Architecture](03-architecture.md#the-mask-is-not-a-grid).

The rewrite is numerically identical: the reference logits from before the change
reproduce exactly (`max_abs_diff=0.000e+00`), because the attention pattern is
unchanged — only how the mask is expressed.

## What is still on the table

Two measured inefficiencies remain. Neither is a capacity problem; both are work
we do twice or do on filler.

**1.41× of attention work is padding.** Batching to the longest row in the batch
means shorter rows are processed at that length. Measured on the S1MB test split
with `max_rows=256`, padded tokens are 2.18M against 1.54M real.

**2.5× of the state tower is redundant.** An example expands into one row per
candidate, and every one of those rows carries the same state. Because the state
cannot see the query (prefix-LM, [Architecture](03-architecture.md)), its hidden
states and its K/V are *identical* across those rows — yet each is computed
separately. Rows per case: p50 1, p99 28, and one case in the corpus has 2,416
candidates.

Two changes would take them, in this order:

- **Shared prefix.** Compute the state tower once per case, then reuse its K/V
  across that case's candidate rows. Exact, not an approximation, and it removes
  the 2.5× (for the 2,416-candidate case, essentially all of it). This is the
  same idea as KV caching — the account is "don't compute identical work twice",
  just across candidates instead of across decode steps.
- **Length bucketing.** Group rows by exact state length so no state padding
  exists. That removes the 1.41× *and* the remaining mask: with no padding there
  is nothing to exclude, so the query call can use `is_causal=True` (0 bytes
  instead of 30 MB, 28 ms instead of 52 ms). The cost is more, smaller batches
  when lengths are scattered — the long states all sit exactly at the 192 cap and
  bucket perfectly, but short states fragment.

Neither is done yet. Bucketing in particular needs a real measurement before it
is trusted: fewer rows per batch means more kernel launches, and that can easily
cost more than the padding it saves at these sequence lengths.



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

## First clean run (2026-11-15)

The first run after the encoding fix, and the first number in this project that is
comparable to the published table.

```
BEKKO_CASES=300 SYSTEMONE_EPOCHS=1   (query budget 64, 89,442 examples, 2,796 steps)
```

Raw accuracy moved as the diagnosis predicted, and only on `choice`:

| metric | before fix | after fix |
|---|---:|---:|
| train `choice` | 0.418 | **0.462** |
| test `choice` | 0.308 | **0.362** |
| test `noul` | 0.698 | 0.677 |
| test `score` | 0.526 | 0.511 |
| overall | 0.534 | 0.539 |

`choice` is the one kind the bug damaged, and it is the only one that improved.
`noul` and `score` moved down slightly, which is expected noise from a fresh
1-epoch run rather than a regression.

**The adjusted number, which is the one that counts:**

```
noul    task score =  2.71
choice  task score =  0.40
score   task score =  0.00
Task Avg          =  1.04     (published: laya 13.36, bekko-17m 27.57, jev 59.59)
```

Zero benchmarks were excluded, so this is the full 137.

What that says:

- **`score` is at exactly zero across all 21 benchmarks.** The model's expected
  value is no better than always guessing the median. This is the clearest
  remaining gap and it is not a capacity question — it is that the model has no
  sense of scale at all.
- **`choice` clears the baseline on 6 of 57 benchmarks**, by a few points each
  (`contract_nli` 0.097, `email-selection` 0.092, the rest under 0.02). It is
  learning something, not much.
- **`noul` clears the baseline on 15 of 59 benchmarks.** Best absolute gains are
  on `tracie` (0.120) and `argument_quality` (0.044).

A Task Avg of 1.04 is far below even Laya's 13.36. That is the honest starting
point, and it is now measurable — which it was not before.

## What the results imply for next steps

1. **Train longer.** This was 1 epoch on 300 cases/subset. The old "0.534" came
   from 2 epochs on 1000/subset. Nothing here indicates convergence; `train_acc`
   on `choice` was still climbing.
2. **Fix `score`.** A flat zero across 21 benchmarks suggests the expected-value
   head is not learning the scale, not that the task is hard. Worth checking
   whether the level values reach the model at all and whether 1 epoch is simply
   too few for 13,889 examples.
3. **Then test capacity.** 1M → 4.2M is the supported next move; the
   17M/68M/400M bekko scaling (27.57 / 40.46 / 50.60 Task Avg) suggests headroom,
   but "the 1M model is capacity-bound" is still a hypothesis.
4. **Distillation** is the only lever that adds information rather than
   reweighting what we have. Laya is Apache-2.0 and runnable.

## Known gap: `ranking` decisions are dropped

The bekko corpus contains a fourth decision type, `ranking`, which has
`type: null`, no `criteria`, and its candidates in `documents`. The parser drops
every one — `agnews` alone is 19,992 decisions, an entire subset. `S1MB` contains
no `ranking` decisions, so scores are unaffected; this is training data left on the
table. The drop warning now names it (`unsupported:ranking`) instead of printing
an opaque `None`.

Next: [Code map →](12-code-map.md)
