# 06 — Training

What is optimised, how, and how to reproduce it. File:
[`train.py`](../src/systemone/train.py).

## The loss is per question, not per row

One batch contains many groups. For each group the loss is computed over that
group's rows, and the group losses are averaged:

```python
total = 0
for group in groups:
    rows = (group_ids == group).nonzero()
    if kind == "noul":
        total += BCE(logit, target)              # one row
    elif distribution is not None:
        total += -sum(weights * log_softmax(rows))   # soft label
    else:
        total += cross_entropy(rows, target)     # hard label
return total / len(groups)
```

Three consequences worth understanding:

1. **Averaging over groups means every question counts equally**, regardless of
   how many candidates it has. A 2-option question and an 8-option question
   contribute the same. That is deliberate, and it also means a `score` group
   with 4 levels does not outvote a `noul` group.
2. **Soft labels are first-class.** If the dataset says "70% B, 30% C", the loss
   is the cross-entropy against that distribution. This is strictly more
   information than `argmax`, and it is why calibration improved on soft-label
   runs.
3. **`noul` uses BCE on a single logit**, everything else uses softmax across the
   group's rows.

In code, `EncodedSet.collate` precomputes `noul_mask`, `row_targets`, and
`group_sizes` on CPU, and `batch_loss` computes the same quantity in a single
vectorised `float32` pass (`scatter_reduce_` + `scatter_add_`) with no per-group
Python loop or GPU→CPU synchronisation.

## Accelerators (CUDA, MPS, CPU)

`default_device()` selects `cuda` > `mps` > `cpu`. When reduced precision is
enabled (`bf16=True`, which `scripts/train_main.py` turns on by default on CUDA):

- **CUDA with native `bf16`** (L4, A100, H100) and **MPS / CPU** use `bfloat16`.
- **CUDA without native `bf16`** (Colab/Kaggle T4, P100) automatically falls
  back to `float16` with `GradScaler`, so it hits Tensor Cores instead of slow
  emulated `bf16`.
- On CUDA, `train()` also enables TF32 matmuls (`set_float32_matmul_precision("high")`),
  uses non-blocking host-to-device copies, and supports `torch.compile` via
  `compile=True` (`SYSTEMONE_COMPILE=1`).

## The optimiser and schedule

```python
AdamW(lr=2e-3, weight_decay=0.01)
clip_grad_norm_(1.0)
warmup  = 5% of total steps, linear 0 → lr
cosine  = lr → 5% of lr
```

With `epochs × ceil(len(dataset)/batch_size)` steps known up front, the schedule
is a `LambdaLR`. Warmup matters because a 2e-3 LR from step 0 on a freshly
initialised encoder can produce one huge update that ruins the embeddings;
cosine decay is standard for small models where the last few hundred steps
matter.

`History` records `losses` (one per step), `lrs` (one per step), and `evals`
(epoch, eval loss) when an `eval_dataset` is passed.

## A training step, literally

```python
batch = dataset.collate(chunk).to(device)     # int32 IDs, padding per batch
logits = model(batch.state_ids, batch.state_mask,
               batch.query_ids, batch.query_mask)
loss = batch_loss(logits, batch)
optimiser.zero_grad(set_to_none=True)
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimiser.step()
scheduler.step()
```

No dropout, no label smoothing beyond what the data already carries, no EMA.

## Batching: rows, not examples

A batch is capped by **rows**, because an example expands into one row per
candidate and a test choice can carry thousands. `EncodedSet.batches(max_rows)`
and `train._chunks` both enforce that cap.

Within the cap, order decides how much padding is paid. A batch pads every row
to its longest state and longest query, so a natural-order batch that happens to
hold one 946-token example makes every 20-token example in it pay for 946 tokens.
At 1024/320 on real fit-only rows, that came to **10.4×** the real state tokens.

`EncodedSet.bucketed_order` groups examples into 16-token length bands and
shuffles inside each band. Bands keep a batch's padded shape close to its own
members; the shuffle keeps membership random, so no batch becomes
label-homogeneous. That is measured at **2.77×** on the loop and documented with
the numbers in [Experiments](11-experiments.md#bucketing-by-length-rejected-at-192-adopted-at-1024).

Evaluation sorts by length unconditionally — it has no gradient and no order
dependence — while `train(..., bucket=False)` restores the plain shuffle.

## Reproducibility

```python
set_seed(seed)          # random, numpy, torch, cuda
model = build_model(cfg, seed)   # seed before construction
```

The seed is stored in `model.pt`. Two caveats:

- **MPS is not bit-reproducible.** Apple's GPU kernels can differ run to run. For
  byte-exact reruns use CPU (`PYTORCH_MPS=0`) or accept small drift.
- **SentencePiece training is deterministic for a fixed corpus**, which is why
  the cache fingerprint is stable across reruns.

## What actually happens in a run

From the measured runs on this machine (MPS):

| run | examples | batch | steps/epoch | seconds/epoch |
|---|---:|---:|---:|---:|
| test slice, 6 epochs | 1,947 | 32 | 61 | ~66 |
| all 106 test configs, 10 epochs | 21,148 | 32 | 661 | ~249 |
| bekko corpus, 2 epochs | 284,877 | 64 | 4,452 | ~3,670 |

The model is small; the data is the wall clock.

## Evaluation is part of the wall clock

A training step and an evaluation step are not symmetric: evaluation has no
backward pass, so it runs about **4x faster per example**. That makes it easy to
assume eval is nearly free. It is not, because it used to be paid several times
over, on a split much larger than the training set.

What `scripts/train_main.py` used to do per epoch, training on a `BEKKO_CASES=300`
slice (3,126 examples) against a 14,903-example held-out split:

| pass | examples | cost |
|---|---:|---|
| evaluate before training | 14,903 | ~37 s |
| evaluate each epoch | 14,903 | ~37 s |
| evaluate after training | 14,903 | ~37 s |
| `collect_logits` on test, twice | 14,903 | ~33 s each |

Training that epoch took 32 s. **Evaluation was the majority of the run**, and
one of the two `collect_logits` passes was redundant with the `evaluate` next to
it — both were computed from the same forward pass over the same examples.

Three rules now:

1. **Measure on no more data than you train on.** The per-epoch probe is capped at
   the training-set size and spread across configs rather than taken as the first
   N. Reporting a loss on 5x the training data is not more honest, just slower.
2. **Score the held-out split once, at the end.** There is no "before" number: an
   untrained model's loss is not something anyone acts on.
3. **One pass, many metrics.** `collect_logits` returns the logits that both
   `collected_loss` and the calibration need, so `evaluate` no longer walks the
   split a second time for a number already in hand.

`collect_logits` also now takes `bf16` and walks length-ordered batches. It was
the only evaluation path still in fp32 and natural order.

## Fast iteration

A 2-epoch pass over the full corpus is roughly 3,700 s per epoch, which makes it
useless as a loop. `SYSTEMONE_DEV=1` shrinks data, configs, and candidates:

```
configs 5          200 cases each       max_candidates 4
rows 128           bf16 on              same eval policy as a full run
name <name>_dev
```

The `<name>_dev` suffix matters: it keeps the dev run away from the real
checkpoint and its caches. The tokenizer and both dataset caches are reused when
present, and `SYSTEMONE_RESUME=1` continues from the last saved epoch — optimizer
state and LR schedule included — instead of starting over.

Measured: a 4-epoch dev run including dataset load and encoding is a few minutes;
resuming it for four more epochs is 47.5 s of training.

**Caveat, from the first dev runs.** At 685 training examples the model overfits
inside two epochs — eval loss 1.288 → 1.296 → 1.381 → 1.460, and it keeps
climbing. So the dev profile is a *pipeline check*: it tells you a change trains
at all and produces a curve, not whether the change is better. Any conclusion
drawn there has to be confirmed on a larger slice before it is believed.

## Known limitations of the current recipe

These are deliberate, documented gaps rather than hidden bugs:

- **No per-kind weighting.** `noul`, `choice`, and `score` are equal per group,
  but their example counts are not equal (~50/34/16% in the bekko corpus).
- **No curriculum.** Instructions and criteria are sampled uniformly, not from
  easy to hard.
- **Overfitting is not fought.** No early stopping, no dropout. Eval loss is
  logged and you read it.

Next: [Inference →](07-inference.md)
