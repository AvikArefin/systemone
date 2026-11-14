# 10 — Metrics

Why the leaderboard number is not "accuracy", and what it takes to produce a
comparable one. Spec: S1MB `evaluator/SCORING.md`.

## Raw accuracy is not enough

`accuracy_by_kind()` in [`calibrate.py`](../src/systemone/calibrate.py) gives the
plain fraction correct. Useful for debugging. Not comparable to published S1MB
scores, because:

- `noul` classes are imbalanced → always answering "false" scores well.
- `choice` option counts differ per question → 2-way and 77-way questions are
  mixed.
- `score` is ordered → predicting the middle every time scores decently.

S1MB therefore reports **baseline-adjusted** scores: 0 means "no better than the
trivial strategy", 100 means "at the ceiling".

## The three adjustments

### noul — balanced accuracy

With soft targets `t`:

```
predicted true :  TP += t,  FP += 1-t
predicted false:  FN += t,  TN += 1-t

recall_true  = TP / (TP + FN)
recall_false = TN / (TN + FP)
BA           = (recall_true + recall_false) / 2
adjusted     = 2·BA − 1
```

`BA = 0.5` → `0`. Always-true and always-false both score zero. Eligibility: both
classes must have positive target mass.

### choice — versus the best trivial policy

Baseline `b` is the best of:

1. **uniform** — `mean(1 / candidate_count)` per question;
2. **always pick option ID X** — best over all IDs seen in the benchmark;
3. **always pick position P** — best over all positions.

A missing ID/position falls back to uniform. Then:

```
adjusted = (accuracy − b) / (1 − b)
```

Eligibility: `b < 1`. If a trivial policy is perfect, there is no headroom and
the benchmark is excluded for everyone.

Accuracy itself is the **target probability at the selected option** — so soft
labels give partial credit, and argmax ties resolve by original order.

### score — expected-value error

Normalise each level's numeric value to `[0, 1]`, then:

```
predicted_ev = Σ p_i · u_i
target_ev    = Σ t_i · u_i
MAE          = mean |predicted_ev − target_ev|
baseline     = median(target_ev)              # best constant
baseline_MAE = mean |target_ev − baseline|
adjusted     = 1 − MAE / baseline_MAE
```

This is why the model outputs an expected value per level rather than a class
label.

## Aggregation

```
per benchmark:  average decisions → adjusted skill → clip to [0, 1]
per task:       equal-weight mean of eligible benchmarks
Task Avg:       equal-weight mean of the three task scores × 100
```

Clipping happens **per benchmark** and is why a catastrophically bad subset cannot
sink the whole score. The trade-off is that it also hides how far below baseline
a model is — which is why raw metrics stay visible in the results files.

`Borda Score` is a different, purely relative number: rank every model per
benchmark, award `100·(N−r)/(N−1)` points, average over all 137 benchmarks. It
depends on the model roster, so it is not a property of a model in isolation.
"Beat everyone" is a Borda statement; "is the model useful" is a Task Avg
statement.

## What this project can and cannot report today

`scripts/train_*.py` prints **raw accuracy per kind** plus ECE. Nothing in this
repo yet computes the adjusted scores, which means:

- Our internal numbers (0.53, 0.55) are raw accuracy on a case-split.
- Published numbers (Jev 59.59, bekko-400m 50.60, Laya 13.36) are Task Avg.
- **They must not be compared directly.**

Producing a comparable number needs four things, all straightforward:

1. Load S1MB with `max_candidates=None, shuffle_options=False` so the benchmark's
   own candidate sets and order are used.
2. Run the model, collect per-decision predicted distributions.
3. Apply the three formulas above, grouped by `(subset, kind)`.
4. Emit per-benchmark table, task scores, and Task Avg.

The `Example` dataclass already carries `benchmark`, `candidate_ids`, and
`values`, which is exactly the metadata those formulas need — the loader was
generalised for this, but the scoring script is not written yet.

Next: [Experiments →](11-experiments.md)
