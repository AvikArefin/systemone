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

Padding is removed by ANDing with `key_mask`, so padded positions can never be
attended to.

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
