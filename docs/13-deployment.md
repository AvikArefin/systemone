# 13 — Deployment

Why the model is 1M parameters, and what has to happen before it runs on a chip.

## The budget

The target is a microcontroller with **8 MB of PSRAM**. Model weights must fit
alongside the runtime, the tokenizer, and activations. Sizing the weights:

| Configuration | Params | fp32 | int8 | ternary (~1.6 bit) |
|---|---:|---:|---:|---:|
| deployed (`d_model=128`, 4 layers) | 1,001,472 | 4.0 MB | 1.0 MB | 0.2 MB |
| default (`d_model=256`, 6 layers) | 4,225,024 | 16.9 MB | 4.2 MB | 0.85 MB |

So:

- The **1M** model fits in float32 with room to spare.
- The **4.2M** model *only* fits quantised — int8 is 4.2 MB, which is roughly half
  the PSRAM and leaves the rest for activations and buffers. **4.2M is the
  ceiling**, not a waypoint to 10M.

That number is the reason the capacity conversation in
[Experiments](11-experiments.md) stops at 4.2M.

## What else costs memory

**Activations.** Per row, per layer: a hidden vector of `d_model` floats plus
attention scratch. With `d_model=128` and a 192+48 token sequence this is tens of
KB, not MB. Fine.

**KV cache.** `ModelConfig.kv_cache_bytes(seq_len)` =
`2 × layers × kv_heads × head_dim × seq_len` elements. For the deployed config at
240 tokens: `2 × 4 × 2 × 32 × 240 = 122,880` elements = 492 KB in fp32, 123 KB in
int8. GQA is what keeps this small — with 4 KV heads instead of 2 it would double.

**Tokenizer.** The SentencePiece model is ~250 KB on disk; the runtime only needs
the vocabulary and merge table. A 2,048-entry vocab of short strings is tens of
KB.

**Embedding table.** 2,048 × 128 = 262k parameters, 26% of the model. Quantising
it to int8 saves ~0.75 MB at some quality cost. Shrinking the vocabulary saves
proportionally more but lengthens sequences, which increases attention cost
quadratically. This is the main sizing trade-off.

## Inference cost

Per forward pass over `R` rows and `L` total tokens:

```
attention  O(R · L² · d_model)      # candidates do not attend to each other
FFN        O(R · L · ffn_hidden)
```

The `R · L²` term dominates. Two consequences for a device:

1. **Truncation is a performance knob, not just an accuracy one.** Halving the
   state length quarters attention cost.
2. **Batching candidates is nearly free** compared with running each candidate
   separately — the state pass is shared in practice, and the candidate branches
   are short.

There is no incremental state cache today: each request re-encodes its state.
Adding one would help only if the same state were queried repeatedly, which is
the common agent pattern — a known future optimisation, not a current feature.

## What has to be written for the device

Nothing in this repo emits chip code yet. The pieces:

1. **Export.** Convert `model.pt` to a static graph (TorchScript/ONNX), then to
   C. SentencePiece also needs a small C runtime or a pre-tokenised input path.
2. **Quantisation.** Post-training int8 for weights plus int8 or fp16 activations;
   ternary needs quantisation-aware training to avoid a large accuracy drop. The
   histogram of weights matters more than the format.
3. **Kernels.** Hand-written attention and matmul for the target ISA. The ESP32
   has vector extensions but no PyTorch — this is the actual engineering work.
4. **Memory plan.** Weights in PSRAM, activations on the stack, streaming the
   embedding lookup rather than copying the table.

Until then, the honest statement is: **the model is sized for the target and
runs on a laptop**; the embedded runtime is future work. Every design choice in
this repo (GQA, RMSNorm, prefix masking, bounded output space) exists to make
that runtime tractable.

## Sizing summary

| Decision | Value | Why |
|---|---|---|
| Parameters | 1M | fits fp32 in 8 MB, leaves room to grow to 4.2M quantised |
| Layers | 4 | latency scales linearly; 4 is the floor for useful depth |
| `d_model` | 128 | 4 heads × 32 is the smallest sane head width |
| KV heads | 2 | halves KV memory and bandwidth |
| Vocabulary | 2,048 | keeps sequences short without eating the budget |
| Output | bounded distribution | no generation, no parsing, no runaway cost |

Back to [docs index](README.md).
