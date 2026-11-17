# 10 — Metrics

Why the leaderboard number is not "accuracy", and what it takes to produce a
comparable one. Spec: the benchmark's `evaluator/SCORING.md`.

## Raw accuracy is not enough

`accuracy_by_kind()` in [`calibrate.py`](../src/systemone/calibrate.py) gives the
plain fraction correct. Useful for debugging. Not comparable to published benchmark
scores, because:

- `noul` classes are imbalanced → always answering "false" scores well.
- `choice` option counts differ per question → 2-way and 77-way questions are
  mixed.
- `score` is ordered → predicting the middle every time scores decently.

The benchmark therefore reports **baseline-adjusted** scores: 0 means "no better than the
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

`scripts/train_*.py` prints **raw accuracy per kind** plus ECE, which is a
debugging signal, not a leaderboard number. `scripts/eval.py` computes the
adjusted scores described above:

```bash
SYSTEMONE_EVAL_CHECKPOINT=bekko_tiny uv run python scripts/eval.py
```

It keeps the benchmark's own candidate sets and order
(`max_candidates=None, shuffle_options=False`) and fails loudly rather than
dropping a decision, so the number it prints is comparable to the published
table. Output goes to `artifacts/<name>/test_report.json`.

All four steps are implemented:

1. Load the test split with `max_candidates=None, shuffle_options=False` so its
   own candidate sets and order are used.
2. Run the model, collect per-decision predicted distributions
   (`scoring.probabilities_by_group`).
3. Apply the three formulas above, grouped by `(subset, kind)`
   (`scoring.build_report`).
4. Emit per-benchmark table, task scores, and Task Avg.

The `Example` dataclass carries `benchmark`, `candidate_ids`, and `values`, and
`EncodedSet` now persists them, so the scoring code can group by benchmark
without re-parsing the dataset.

## Why raw accuracy flatters this benchmark

Measured on the test split, the trivial policies already score:

| kind | best trivial policy | accuracy |
|---|---|---:|
| `noul` | always answer false (71.5% of labels) | 0.715 |
| `choice` | always pick position 0 | 0.254 |
| `score` | always pick the modal level | 0.501 |

A model reporting "0.534 overall" on a 47/33/20 mix is therefore reporting
mostly the label marginals. The adjusted score subtracts that:
the same checkpoint scores `noul` +0.14, `choice` +0.04, `score` −0.32, for a
Task Avg around 6. Quote the adjusted number or say "raw accuracy" explicitly.

Next: [Experiments →](11-experiments.md)
