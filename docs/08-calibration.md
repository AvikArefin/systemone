# 08 — Calibration

A model can rank answers correctly and still lie about its confidence. This page
explains the fix used here. File:
[`calibrate.py`](../src/systemone/calibrate.py).

## The problem

Suppose the model says "0.9" on 100 questions. If it is right on 90 of them, it
is calibrated. If it is right on 60, it is **overconfident** — and any downstream
policy of the form "act automatically above 0.9" is unsafe.

Accuracy says nothing about this. A model can be 70% accurate and wildly
overconfident, or 70% accurate and perfectly calibrated.

## ECE — turning that into one number

Expected Calibration Error:

1. Bucket predictions by stated confidence into 10 bins (`0.0–0.1`, …,
   `0.9–1.0`).
2. For each bin, compare average confidence with average accuracy.
3. Average the gaps, weighted by bin size.

```
ECE = Σ (n_b / N) × |accuracy_b − confidence_b|
```

0 is perfect. 0.05 is good. 0.15 means the stated numbers are decorative.

Implementation detail: for `noul` the confidence is `max(p, 1-p)`, since a
confident "no" is a confidence too. For `choice` and `score` it is the top
probability.

## Temperature scaling — the fix

Divide the logits by a single scalar `T` before the softmax:

```python
probs = softmax(logits / T)
```

- `T > 1` softens probabilities (fixes overconfidence).
- `T < 1` sharpens them.
- `T = 1` changes nothing.

Crucially, dividing all logits of a group by the same positive number **cannot
change the argmax**. The answer stays the same; only the confidence moves. That
is why this is safe to fit after training and store in the checkpoint, and why it
needs no retraining.

## How `T` is fitted

```python
train_collected = collect_logits(model, train_data)   # logits + targets, per kind
temperatures = fit_temperatures(train_collected)
ece_after = expected_calibration_error(test_collected, temperatures)
```

- Collect logits for the **train** split, grouped by kind.
- Search 40 values of `T` geometrically spaced from 0.25 to 10.0.
- Pick the `T` minimising NLL (BCE for `noul`, cross-entropy otherwise).
- Fit one `T` per kind, because the three heads behave differently.
- Measure ECE on the held-out split, never on the one `T` was fitted to.

Fitting on the evaluation split is leakage: it reports calibration the model was
optimised for, and it is a one-parameter fit so the inflation is small but free.
The training scripts previously did exactly that — `fit_temperatures(collected)`
where `collected` came from `test_data`. They now pass `train_collected`, and
`tests/test_regressions.py` asserts the scripts cannot regress.

No gradient descent — a grid is enough for one parameter per kind, and it cannot
diverge.

## What we actually measured

On the 26k S1MB-slice run:

```
ECE before   0.117
ECE after    0.069
T fitted     noul 1.66  choice 2.20  score 3.88
```

All three temperatures are above 1, i.e. the model was overconfident on every
task. The `score` head was the worst offender — its logits are the least
constrained, and with only ~5k score examples it is the least trained.

After the cleaner corpus run:

```
ECE before   0.122   →   after 0.090
T            noul 1.82  choice 2.92  score 1.25
```

An important caveat, seen directly in these numbers: **temperature scaling fixes
confidence, not knowledge.** A model that is wrong is still wrong after
calibration. When an earlier run collapsed to chance on `choice`, calibration
could not rescue it; only more data could.

## Where temperatures live

`save_checkpoint(..., temperatures=...)` writes them into `model.pt`, and
`load_checkpoint` returns them. `answer_request` applies them per kind. If you
change the model's training data significantly, refit — a stale temperature can
make a well-calibrated model look broken.

Next: [Datasets and benchmark →](09-datasets-and-benchmark.md)
