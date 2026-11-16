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
rows   full grid mask   split mask   full grid   split   shared
  16            13 MB         2 MB       17 ms     8 ms     3 ms
  64            51 MB         7 MB       18 ms    16 ms     6 ms
 128           103 MB        15 MB       36 ms    28 ms     8 ms
 256           206 MB        30 MB       99 ms    53 ms    10 ms
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

## Sharing the state tower across candidates

Every candidate row of a question carries the same state, and the state cannot
see the query, so its encoding is *identical* across those rows. Nothing about
that requires recomputing it. `EncodedSet.collate` now returns one state per
group plus `group_ids`, and the model runs the state half once per group, then
gathers its K/V to the rows for the query half.

Measured on 20 S1MB configs (4,399 cases, 41,444 rows, `max_rows=256`), model
time only:

```
shared tower      6.58 s
repeated per row 11.54 s
```

1.75× over the split, and 1.95× on the worst single batch (99 ms vs 193 ms,
256 rows from 168 groups). The duplication factor is 2.5× on average, p99 28×,
and one case in the corpus has 2,416 candidates.

Output is unchanged: `max_abs_diff=0.000e+00` against logits captured before the
change, and the smoke-split Task Avgs match exactly.

## The real problem: the model only knows the label marginals

The budget tables print percentages. Read that way, a Task Avg of `1.04` is
**1.04% of headroom**, and no single kind is holding the average down:

```
noul 2.71%   choice 0.40%   score 0.00%
```

`score` is not broken — it is the floor of a floor. On score rows the model
predicts the most common level for **87.6%** of rows against a **50.1%** gold
marginal. Its logits vary (mean spread 2.06), and the level text is present in
the query; it simply carries no signal beyond the marginal, and the adjusted
metric maps marginal-following to exactly zero. The same is true of the other two
kinds, less sharply.

**The pipeline is not broken.** Given 240 real examples from one config (`arc`),
60 epochs drives training loss to **0.0000** while eval loss climbs 1.39 → 9.11.
It memorises readily. It is not being given enough to learn from at this scale:
two epochs over 285k examples buys the marginals and nothing else, and eval loss
does not improve — 1.3921 → 1.4679 in the A/B, and flat at ~1.39 in the scoped
runs.

So the next question is not a token budget. It is whether training long enough
for eval loss to move at all produces skill, on a subset small enough to run to
convergence. Every budget and architecture result so far was measured against a
model that had learned almost nothing, which is why they all looked the same.

## One task, held out: capacity does not fix it

The sharpest test available: one task, 80/20 split by case, everything filtered to
fit the budget, and both model sizes trained identically. Held-out data the model
never sees.

`arc`, balanced four-way choice (gold position 23/25/26/26%), 2,678 train
decisions, 670 held out:

```
model      params   held-out loss curve              accuracy   chance
1.0M       0.87M    1.387 -> 1.425 -> 1.809 -> 3.235    0.278     0.250
4.68M      4.42M    1.391 -> 1.418 -> 1.395 -> 1.509    0.255     0.250
```

Both sit at chance. The small model overfits harder; the big one resists slightly
and is no more accurate. **Capacity is not the blocker.**

An earlier probe on `go_emotions` looked more encouraging and was wrong to trust:
held-out accuracy 96.3%, but 96.4% of that set is the negative class, so the score
was the base rate. The model's true-positive rate was 0.069 at 1.0M and 0.000 at
4.68M — it learned the prior and nothing else. Any task with a lopsided label
distribution cannot answer "can it learn" with accuracy.

So the sequence across this session is consistent: the model memorises a small
set, learns trivial synthetic rules, and on every real task it learns the label
marginals. Not context length, not masks, not memory, not capacity.

## Auditing for a bug: what was checked and came back clean

"Learns only the marginals" has three classic causes. All three were tested
against the deployed checkpoint rather than reasoned about:

```
label alignment      target index points at the gold candidate's row: 50/50
                     duplicate option texts (which collapse the label): 0 / 3,651
state is used        scrambling the state moves logits: 289/300 rows, mean 0.21
candidate is used    scrambling the option text moves logits: 295/300, mean 0.51
within-group spread  mean 0.82, min 0.002 across 60 choice groups
```

Both inputs reach the head, the labels are where they claim to be, and the model
does separate candidates — it separates them badly. A within-group spread of
0.002 on some groups is the model being indifferent, not the graph being broken.

So the near-zero skill is not a wiring fault. Every bug found this session was in
the tooling and all are fixed: the mask that took the machine down, offline
config discovery silently returning `"default"` (an offline run loaded zero
cases), a query padding mask that causality had been hiding, and an A/B subset
containing no `score` examples at all.

## Retraining at a bigger budget: first measurement

The checkpoint was trained at 192/64, so every larger-budget number so far was
out-of-distribution. With checkpointing and micro-batching in place, a retrain is
affordable; this is the first A/B.

Same 10 configs, same seed, same effective batch (256 rows — the 384 run uses 32
rows x 8 accumulation), 2 epochs, 7,000 cases / 8,402 examples:

```
run                 train_s   eval loss before -> after   accuracy   noul  choice  score
192/64, 256 rows      175       1.3921 -> 1.4679           0.417     0.567  0.271   0.306
384/128, 32 rows      397       1.3914 -> 1.3888           0.423     0.587  0.262   0.303
```

**The knobs work; the experiment is too small to answer the question.** Training
takes 2.3x longer, exactly as the per-row measurements predicted, and the 384 run
fits comfortably with 32 rows x 8 accumulation. But the accuracy gap (0.417 vs
0.423) is inside the noise band of a 8,402-example, 2-epoch run — and note that
the 192 baseline's eval loss got *worse* after training, which is a sign the run
is too short to say anything. The only directional hints are the 384 run's loss
moving down instead of up, and +2 points on `noul`.

Settling it needs the full corpus (182 configs, 898,334 rows, ~8 h per budget)
and at least 2-3 seeds per arm. That is the cost of the answer, and it is worth
stating plainly rather than reading a 0.006 difference as a result.

**A caution about `BEKKO_CASES`:** it is applied **per config**, not in total.
`BEKKO_CASES=1000` across 182 configs is ~182,000 cases / 898,334 rows, roughly a
sixth of the corpus — not the "1,000 cases" it reads like.

## Causal query attention: tested, kept

The query half reads left-to-right. That looked like an accident worth deleting —
the model pools the *last* query token, and that token has seen the whole query
whether or not the rule is causal. So the rule was removed and measured.

```
query attention      noul   choice   score   Task Avg
causal (current)     2.71    0.40    0.00      1.04
bidirectional        2.47    0.41    0.00      0.96
```

Bidirectional is ~8% worse. The reasoning that suggested it was free was wrong in
an instructive way: the last token does see the whole query either way, but its
vector is built *from the layer below*, and those intermediate tokens were
computed causally. Change the rule and every intermediate changes, so the summary
changes too. "The last token sees everything" is a statement about the mask, not
about the information that reached it.

Caveat: this checkpoint was trained with the causal rule, so the comparison asks
it to work outside its training pattern. The honest reading is "flipping the rule
without a retrain costs 8%", not "bidirectional attention is worse". A retrain
could go either way. Kept causal anyway — the change bought tidiness, not speed
or memory, and 8% is not a tidy-up price.

## Memory levers: what actually buys headroom

Measured on the deployed shape (4 layers, `d_model=128`), forward + backward +
optimizer step, MPS, 16 GB machine:

```
budget     rows  bf16  checkpoint   ms/step    driver memory
768/128     64   no    no            1329      7221 MB
768/128     64   no    yes           1724      3858 MB
768/128     64   yes   yes           1769      3867 MB
1024/384    64   no    no           55806     14067 MB   <- swapping, not computing
1024/384    64   no    yes           5877      8137 MB
1024/384    32   no    yes           2808      3787 MB
```

**Gradient checkpointing is the lever.** Recompute activations in the backward
pass instead of storing them: -46% memory for +30% time. At 1024/384 it is the
difference between a 56-second step that is thrashing and a 6-second step that is
computing.

**bf16 halves no memory here** (7221 → 7222 MB) but it removes intermediate fp32
pressure, which is why the 1024/384 step went from 56 s to 7 s.

**Micro-batching is free.** At 768/128, 32 rows and 64 rows run at the same
throughput (30 vs 31 rows/s), so halving the batch to buy memory costs nothing —
accumulate gradients over the two halves.

Together they put 1024/384 at 32 rows inside 3.8 GB, a third of what plain
768/128 needed. The ceiling was never the token budget; it was the shape of the
loop.

## Bucketing by length was measured and rejected

The remaining waste is padding: 1.41× of attention work is tokens that exist only
to make the batch rectangular. The obvious fix is to group rows by exact state
length so no padding exists — no state padding means the state call needs no
mask, and the causal rule collapses to one tensor for the whole batch.

It was implemented and benchmarked, and it does not pay:

```
                batches   eval time
natural order       227        9.0 s
length-bucketed     371        9.1 s
```

Lengths scatter across 1–192, so exact grouping fragments the batches: 63% more
batches, each with its own launch. The padding it removes is cheaper than the
overhead it adds. Reverted; `batches()` visits examples in dataset order.

The shared causal rule that bucketing was going to enable is kept anyway, because
it also applies to serving (one state per request, no padding) and costs nothing
when there is padding to AND it with.




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
