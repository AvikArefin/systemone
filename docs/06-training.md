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

## Fast iteration

A 2-epoch pass over the full corpus is roughly 3,700 s per epoch, which makes it
useless as a loop. `SYSTEMONE_DEV=1` shrinks everything except the model:

```
configs 5          200 cases each       max_candidates 4
rows 128           bf16 on              probe eval (2,000 test examples) each epoch
name <name>_dev    full split evaluated once, at the end
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
- **No length bucketing.** Batches are shuffled globally, so a batch can mix a
  20-token and a 190-token state and pad everything to 190. Bucketing would cut
  wasted compute.
- **No curriculum.** Instructions and criteria are sampled uniformly, not from
  easy to hard.
- **Overfitting is not fought.** No early stopping, no dropout. Eval loss is
  logged and you read it.

Next: [Inference →](07-inference.md)
