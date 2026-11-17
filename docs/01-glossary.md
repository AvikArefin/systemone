# 01 — Glossary

Plain definitions of every term this project uses. Skim it once; come back when a
later page uses a word you do not know.

## Data shapes

**Tensor** — a multi-dimensional array. A vector is 1-D, a matrix 2-D. PyTorch's
datatype for these is `torch.Tensor`. Everything the model touches is a tensor.

**Sequence** — a list of tokens in order, e.g. a sentence. The network sees a
`[batch, length]` grid of token IDs.

**Batch** — several examples processed together. A batch of 32 with 179 candidate
rows means 32 examples expanded into 179 rows (see *row* below).

**Row** — one unit of scoring: a state plus one question plus **one** candidate.
For `noul` there is exactly one row per example. For `choice` with 8 options there
are 8 rows sharing the same state and question text.

**Group** — the rows that belong to the same question. The model emits one logit
per row, and a softmax *within the group* turns those logits into probabilities.

## Tokens and embeddings

**Token** — the smallest unit of text the model sees. Here it is a subword piece
produced by SentencePiece BPE, not a whole word.

**Vocabulary** — the fixed list of tokens. Ours has 2,048 entries, plus ID 0 for
padding.

**Token ID** — the integer index of a token in the vocabulary.

**Embedding** — a lookup table from token ID to a vector of `d_model` numbers.
`nn.Embedding(2048, 128)` means 2,048 rows of 128 numbers.

**[PAD]** — filler token (ID 0) added so all sequences in a batch have the same
length. It is masked out so it cannot affect attention, and its embedding row is
kept at zero.

**[SEP]** — separator token that marks the boundary between the question
instructions and the candidate text. Read it as `tokenizer.sep_id`, never as a
literal: the byte tokenizer puts it at ID 1, while SentencePiece reserves ID 1
for `<unk>` and puts `[SEP]` at ID 2.

**`<unk>`** — the token emitted for a character the vocabulary does not contain.
A whole run of unknown characters collapses into a single `<unk>`, so text with
them still encodes, just without detail.

**Tokenizer fingerprint** — a short hash of the vocabulary. Two tokenizers with
the same `vocab_size` produce tensor-compatible weights that mean different
things, and loading weights checks shapes only. The fingerprint is what lets a
checkpoint refuse a vocabulary it was not trained with.

## Network pieces

**Transformer block** — the repeating unit: attention (mixing information between
positions), then a feed-forward network (processing each position), each wrapped
in a residual connection.

**Attention** — the operation that lets each position look at other positions and
pull in their information. It produces three vectors per position: query (Q),
key (K), value (V). See [Architecture](03-architecture.md).

**Head** — one independent attention "channel". With 4 heads, the model looks at
the sequence 4 ways in parallel.

**GQA (grouped-query attention)** — fewer key/value heads than query heads. Ours
uses 4 query heads and 2 KV heads, so each KV head is shared by 2 query heads.
It saves memory with almost no quality loss; it is the trick used by Llama-class
models.

**RMSNorm** — a cheap normalisation layer that rescales each vector to a sane
magnitude. `torch.nn.RMSNorm` is used directly.

**RoPE (rotary position embedding)** — encodes position by rotating Q and K
vectors by an angle proportional to position. This is how the model knows word
order without a learned position table.

**SwiGLU** — the feed-forward design used here: two up-projections, one gated by
`SiLU`, multiplied together, then projected back down. Standard in modern small
language models.

**Logit** — the raw, unbounded score the model outputs. One number per row.

**Softmax** — turns a group of logits into probabilities that sum to 1.

**Sigmoid** — turns a single logit into a probability in [0, 1]. Used for yes/no.

**Residual connection** — `x + f(x)`; lets gradients flow and makes deep stacks
trainable.

**Parameter** — one learned number (a weight). "1M parameters" means the model
stores ~1 million floats.

## Training

**Loss** — a number measuring how wrong the prediction is; training pushes it down.

**Cross-entropy (CE)** — the loss for "pick one of N". Low when probability mass
is on the right answer.

**Binary cross-entropy (BCE)** — the loss for yes/no.

**Soft label / distribution** — instead of "the answer is B", the target can be
"70% B, 25% C, 5% A". Our `Example.distribution` carries this, and the loss
accepts it.

**Epoch** — one pass over the training data.

**Step / update** — one optimiser update on one batch.

**Learning rate (LR)** — how far each update moves the weights.

**Warmup** — starting with a tiny LR and ramping up, to avoid blowing up early.

**Cosine schedule** — smoothly decaying the LR from its peak down to a floor.

**AdamW** — the standard optimiser; adapts per-weight step sizes and decouples
weight decay.

**Gradient clipping** — capping gradient magnitude so one bad batch cannot wreck
the weights.

**Overfitting** — doing well on training data, badly on unseen data.

**Underfitting** — doing badly on both; the model lacks capacity, steps, or data.

**Seed** — the number that makes random choices reproducible. `set_seed` fixes
Python, NumPy, and torch.

## Evaluation

**Accuracy** — fraction correct. Misleading when classes are imbalanced.

**Balanced accuracy (BA)** — average of recall for each class; 50% means chance
for a balanced binary problem.

**ECE (expected calibration error)** — average gap between stated confidence and
actual accuracy. 0 is perfect; 0.07 is decent.

**Temperature scaling** — dividing logits by one learned scalar to soften or
sharpen probabilities. Fixes calibration without changing the ranking.

**Baseline-adjusted score** — a score expressed as "fraction of the gap from a
trivial baseline to the ceiling". Used by the S1MB leaderboard; described in
[Metrics](10-metrics.md).

## Hardware

**MPS** — Apple's GPU backend for PyTorch. Fast, but not bit-reproducible.

**PSRAM** — extra RAM on microcontrollers (8 MB on the target board).

**Int8 / ternary** — storing weights in 8 bits, or in {-1, 0, +1}, to fit a small
device.
