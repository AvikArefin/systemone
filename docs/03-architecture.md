# 03 — Architecture

This page is the network itself: what goes in, what comes out, and what happens
in between. File: [`src/systemone/model.py`](../src/systemone/model.py).

## The contract

```python
logits = model(state_ids, state_mask, query_ids, query_mask)
```

| Tensor | Shape | Meaning |
|---|---|---|
| `state_ids` | `[B, S]` | token IDs of the shared state, padded to `S` |
| `state_mask` | `[B, S]` | `True` for real tokens, `False` for padding |
| `query_ids` | `[R, T]` | token IDs of instructions + **one candidate**, padded to `T` |
| `query_mask` | `[R, T]` | `True` for real tokens |

`B` is the number of examples in the batch, `R` the number of **rows** — one per
candidate. If 32 examples contain 179 candidates between them, `R = 179`. The
state for a row is looked up by `group_ids`, which is why `B` and `R` differ.

Output: `[R]` — one logit per row. Turning those into a distribution is *not*
inside the model; it is a softmax over the rows of one group
(see [Inference](07-inference.md)).

## Walkthrough with real numbers

Using the deployed configuration (`d_model=128, n_layers=4, n_heads=4,
n_kv_heads=2, ffn_hidden=352, vocab=2048`), a batch of 32 examples expands to
179 rows:

```
state_ids   [32, 192] int32
query_ids   [179, 43] int32      # width is the widest row in THIS batch
        │
        ├── Embedding(2048 → 128)      → [32, 192, 128] and [179, 43, 128]
        │
        └── concatenate per row        → [179, 235, 128]
                     │                    └ 192 state + 43 query positions
                     │
             4 × Block(hidden 128, 4 heads of width 32, KV width 2×32)
                     │
                   RMSNorm
                     │
             take the LAST real query position   → [179, 128]
                     │
              Linear(128 → 1)                   → [179] logits
```

## The mask is the interesting part

The model is a **prefix-LM**, not a plain causal transformer. Its attention
pattern, for 3 state tokens and 2 query tokens:

```
            key positions
              s0  s1  s2  q0  q1
        s0 [  ✓   ✓   ✓   ·   ·  ]
        s1 [  ✓   ✓   ✓   ·   ·  ]
query   s2 [  ✓   ✓   ✓   ·   ·  ]
        q0 [  ✓   ✓   ✓   ✓   ·  ]
        q1 [  ✓   ✓   ✓   ✓   ✓  ]
```

- The **state** is read bidirectionally: every state token sees every state token.
- A **query** token sees the whole state plus the query tokens up to itself
  (causal).
- Nothing in the state can see the query. That is what makes the state encoding
  independent of which question is being asked.

Padding is removed by `key_mask`, so padded positions can never be attended to.

### The mask is not a grid

The pattern above is described as a table, but it is never built as one. A
`rows × tokens × tokens` boolean tensor is 206 MB at 256 rows and 896 tokens,
and it is mostly the same value repeated; building it per forward is what
crashed the machine twice (see
[Experiments](11-experiments.md#the-attention-mask-was-the-thing-that-crashed-the-machine)).

Instead each block makes **two attention calls**, one per query set:

| call | queries | keys | mask |
|---|---|---|---|
| state | state | state | `[rows, 1, 1, s]` — one flag per key |
| query | query | state + query | `[rows, 1, q, s+q]` — causal rule AND the same flag |

The causal rule depends only on relative position, so the query mask costs
`q × (s+q)` per row instead of `(s+q)²`. Neither call materialises a grid.

`is_causal=True` alone cannot be used here: padding sits *inside* the prefix a
query attends to, and SDPA does not accept `is_causal` and `attn_mask`
together. The state half is bidirectional, so a causal flag would also be
wrong for it. Splitting by query set is what buys the same saving without
changing the attention pattern.

Measured with `scripts/measure_attention_memory.py` at 256 rows, state 768,
query 128 (one attention call):

```
strategy      mask tensor   attention
full grid         206 MB        89 ms
split (ours)       30 MB        52 ms
is_causal           0 MB        28 ms   (no padding handling; not applicable)
key-pad only        0 MB        57 ms   (no causality; not applicable)
```

### Why padding needs a mask, and causality does not

A batch is a rectangle, so short rows get filled up to the longest row in the
batch:

```
row 0: [ 40 real tokens ][30 PAD]
row 1: [ 70 real tokens          ]
row 2: [ 70 real tokens          ]
```

Attention is a weighted sum — nothing in the arithmetic knows the last 30 values
are filler. They have to be excluded explicitly, and that exclusion is the
padding mask. Causality is different: "token *i* may look at keys ≤ *i*" follows
from position alone, so the kernel can derive it from a flag and never needs a
tensor.

So there are two ways to get rid of the mask: express the rule (a flag), or
remove what the rule cannot express (the padding). Grouping rows by exact state
length does the second — no PAD tokens exist, so there is nothing to exclude, and
the query call becomes `is_causal=True` with a zero-byte mask. We do not do this
yet: see [Experiments](11-experiments.md#what-is-still-on-the-table).


## Inside one block

```
x ──► RMSNorm ──► Attention ──► (+) ──► RMSNorm ──► SwiGLU ──► (+) ──► out
 └──────────────────────────────┘        └──────────────────────────┘
        pre-norm residual                    pre-norm residual
```

**Attention.** Four projection matrices, no biases:
`q_proj` 128→128, `k_proj` 128→64, `v_proj` 128→64, `o_proj` 128→128.
The 64 is `n_kv_heads × head_dim = 2 × 32` — **grouped-query attention**. Four
query heads share two key/value heads, halving KV work. `enable_gqa=True` on
`F.scaled_dot_product_attention` does the broadcast; PyTorch picks the fastest
kernel available (SDPA on CPU/MPS, FlashAttention on supported CUDA).

**RoPE.** Positions are applied by rotating Q and K, not by adding a position
vector. Note a deliberate detail: the state uses positions `0..S-1` and the query
independently uses `0..T-1`. Each segment has its own coordinate frame, which is
what you want when they are separate texts stitched together.

**SwiGLU.** `down(silu(gate(h)) * up(h))` — two 128→352 projections, one passed
through SiLU and used as a gate on the other, then 352→128.

**RMSNorm.** `torch.nn.RMSNorm(128, eps=1e-6)`. Two per block (attention norm and
FFN norm), plus one final norm outside the stack.

## Where the parameters go

`ModelConfig.parameter_count()` **measures** the count by building the model on
PyTorch's `meta` device, which materialises the full parameter graph without
allocating storage. There is deliberately no hand-written formula: an earlier
version had one, and it silently over-reported by 768 parameters by counting
three norms per block instead of two. Measured, it cannot drift from the model.

The per-component breakdown for the deployed model is still worth understanding:

```
embedding   2048 × 128                  =  262,144
attention   (4+2·2)·32·128 + 128²       =   49,152   per layer
  ├ q_proj  128 × (4 × 32)              =   16,384
  ├ k_proj  128 × (2 × 32)              =    8,192
  ├ v_proj  128 × (2 × 32)              =    8,192
  └ o_proj  128 × 128                   =   16,384
feed-forward 3 × 128 × 352              =  135,168   per layer
norms       2 × 128                     =      256   per layer
                                        ─────────
per layer                               =  184,576   × 4 = 738,304
final norm + head                       =      256
                                        ─────────
total                                   = 1,000,704  (~1.00M)
```

Note that **26% of all parameters are the embedding table**. That is a real
trade-off: a bigger vocabulary shortens sequences but eats the parameter budget.
See [Deployment](13-deployment.md).

The default `ModelConfig` (`d_model=256, n_layers=6, n_heads=8, ffn_hidden=688,
vocab=258`) is 4,222,976 parameters — the largest that still fits the memory
target after quantisation.

## Padding and pooling

Pooling takes the **last real query token** of each row, using
`query_mask.sum(dim=1) - 1`. With `instructions ␟ candidate`, that position sits
at the end of the candidate text, so the pooled vector has seen the state, the
instruction, and the candidate. A single `Linear(128→1)` turns it into the row's
logit.

## Why there is no softmax here

The model returns raw logits. The choice between candidates is made **outside**,
per group:

```python
group_logits = logits[batch.group_ids == g]
probs = torch.softmax(group_logits / temperature, dim=0)
```

Keeping it outside means the same forward pass supports `noul` (one row,
sigmoid), `choice` (N rows, softmax), and `score` (N levels, softmax then
expected value), and lets calibration change the temperature without touching
weights.

Next: [Tokenizer and prompts →](04-tokenizer-and-prompts.md)
