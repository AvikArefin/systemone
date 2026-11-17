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
- Pick the `T` that gives the lowest loss (BCE for `noul`, cross-entropy otherwise).
- Fit one `T` per kind, because the three heads behave differently.
- Measure ECE on the held-out split, never on the one `T` was fitted to.

`calibrate.py` itself never sees a split — it fits whatever logits you pass it. The
choice of split is the caller's. The training scripts used to pass the eval split,
which means the reported ECE included a parameter tuned on that same data. They now
pass the train split.

**How much did that matter? Almost nothing.** Measured on two halves of S1MB:

```
half A: ECE with its own fit  0.0236      with the other half's fit  0.0238
half B: ECE with its own fit  0.0602      with the other half's fit  0.0758
```

One free parameter over thousands of examples does not move much. The fix is still
correct — it is free, and it stops the number from being self-referential — but it
was not a large error, and this page overstated it when the fix landed.

## The bigger problem: `T` is often not really fitted

The temperature is only meaningful if the loss actually responds to it. On a model
that is near chance, it doesn't:

```
choice, same model, two disjoint halves of S1MB:  T = 4.69  vs  T = 1.82
noul, best grid point is the edge:                T = 10.0 (wants ~17.6 at high=1000)
```

Those two `choice` numbers cannot both be right. Look at why:

```
mean NLL   T=1.0 → 0.7697
           T=10  → 0.6929
           T=50  → 0.6929
           T=1000→ 0.6931
```

Past `T≈4` the curve is flat to four decimals, so every grid point up there fits
equally well and the search just returns whichever one it evaluates. That is not a
fitted parameter, it is an artefact of the grid. `fit_temperatures` now warns in
both cases: when the optimum sits on the grid edge, and when the loss barely moves
across the whole grid (`is_identifiable`).

Treat a stored temperature as meaningful only when no warning was printed. If one
fires, the honest answer is "this model's confidence is not recoverable by a single
scalar" — which is a statement about the model, not a bug to fix with a wider grid.

Both edges mean the same thing in different directions:

- **Hits `0.25`** (the floor): the model is *under*confident and wants sharpening
  beyond the grid. A well-trained model on an easy task does this — `train_synthetic.py`
  prints three such warnings at 100% accuracy, which is correct, not alarming.
- **Hits `10.0`** (the ceiling): the model is *over*confident, and the near-flat
  curve means no single `T` fixes it. This is the case to worry about.

No gradient descent — a grid is enough for one parameter per kind, and it cannot
diverge. But a grid cannot rescue a flat objective, so the warning matters more
than the search.

## What we actually measured

On the 26k S1MB-slice run:

```
ECE before   0.117
ECE after    0.069
T fitted     noul 1.66  choice 2.20  score 3.88
```

After the cleaner corpus run:

```
ECE before   0.122   →   after 0.090
T            noul 1.82  choice 2.92  score 1.25
```

**Read these with care.** Every one of these runs used the corrupted encoding, and
given the flat-curve problem above, some of these temperatures are probably grid
artefacts rather than real fits. The ECE improvement is real in direction — all
three values are above 1, so the model was overconfident — but the exact numbers
should be re-derived after a clean retrain.

An important caveat, seen directly in these numbers: **temperature scaling fixes
confidence, not knowledge.** A model that is wrong is still wrong after
calibration. When an earlier run collapsed to chance on `choice`, calibration
could not rescue it; only more data could. The flat loss curve above is the same
fact in another form: if there is no signal, there is nothing to calibrate.

## Where temperatures live

`save_checkpoint(..., temperatures=...)` writes them into `model.pt`, and
`load_checkpoint` returns them. `answer_request` applies them per kind. If you
change the model's training data significantly, refit — a stale temperature can
make a well-calibrated model look broken.

Next: [Datasets and benchmark →](09-datasets-and-benchmark.md)
