# 07 — Inference

What happens when a caller asks one real question. File:
[`inference.py`](../src/systemone/inference.py).

## The interface

```python
answers = answer_request(model, tokenizer, Request(
    state="We were charged twice for the same order.",
    questions={
        "safe":    Noul(instructions="Is it safe to refund without review?"),
        "team":    Choice(instructions="Which team?", criteria=["billing","security","support"]),
        "urgency": Score(instructions="How urgent?", criteria=["week","week","today","now"]),
    },
))
```

`answers` maps each name to a typed answer with probabilities and a confidence.
All questions share one state and are answered in **one batched forward pass**.

## The steps

1. **Truncate the state** to `cfg.max_state_tokens` using head/tail truncation.
2. **Expand every question into rows** — one per candidate, or one for `noul`.
3. **Pad** all rows to the widest row (each row already built to its own budget).
4. **Keep the state once per question.** Every candidate row carries the same
   state, and the state cannot see the query, so its encoding is identical across
   them — the model computes it once and reuses it.
5. **Forward pass** → one logit per row.
6. **Group and normalise** — softmax within each question, after dividing by that
   kind's calibration temperature.
7. **Build typed answers.**

Batches are capped by **rows**, not by examples (`EncodedSet.batches(max_rows)`).
One request can expand into thousands of rows, and the attention mask grows with
rows × tokens², so the example count is not a useful limit. See
[Architecture](03-architecture.md#the-mask-is-not-a-grid).

Serving passes one state with `group_ids = zeros(rows)`, so the tower is computed
once for the request rather than once per candidate.

Two details that are easy to get wrong:

- The state is one buffer repeated for every row. If a request has 3 questions
  with 1, 3, and 4 candidates, that is 8 rows and 8 copies of the same state.
  Sharing the encoding further would need a persistent KV cache, which is not
  implemented (see [Deployment](13-deployment.md)).
- Padding width is the widest row after each row has been built to its own budget.
  `answer_request` calls the same `choice_query` / `score_query` helpers as
  training, so a served request is encoded identically to a training example —
  that equality is asserted in `tests/test_inference.py`, because serving and
  training drifting apart is silent and expensive.

A candidate is **never** dropped to make room for instructions: the row budget is
split with a floor for each half (see
[Tokenizer and prompts](04-tokenizer-and-prompts.md)). An earlier version
truncated the whole row from the front, which deleted the candidate entirely on
long instructions.

## From logits to answers

**`noul`** — one row, one logit:

```python
p = sigmoid(logit / temperature)
```

**`choice`** — N rows for this question:

```python
probs = softmax(logits / temperature)
choice = argmax(probs)
confidence = max(probs)
```

**`score`** — N levels, in ascending order:

```python
probs = softmax(logits / temperature)
score = sum(level_index * p)      # expected value, e.g. 1.7 on a 0-3 scale
```

Note that `score` returns a **fractional** number, not a class. That is the right
output for an ordered scale: "1.7" carries more information than "2", and it is
what the S1MB metric measures (mean absolute error against the expected value).

## Worked example

From `scripts/smoke.py` on an untrained model — the point is the shapes, not the
values:

```
safe    NoulAnswer(noul=0.473)
team    ChoiceAnswer(choice='security', probabilities={billing:0.328, security:0.383, support:0.290}, confidence=0.383)
urgency ScoreAnswer(score=1.507, probabilities={0:0.243, 1:0.241, 2:0.283, 3:0.233})
```

## Temperatures

`answer_request` accepts `temperatures: dict[str, float]`, keyed by
`"noul" | "choice" | "score"`. They come from the checkpoint and are fitted after
training — see [Calibration](08-calibration.md). Passing nothing means 1.0 for
all three.

Temperatures do not change which answer wins. They change how confident the
reported probability is, which is what matters if you use a threshold like
"refund automatically only if p > 0.9".

## Cost model

Per request with `R` rows and `L = state + query` tokens:

```
attention work ≈ O(R × L²)          # candidates do not attend to each other
memory         ≈ 4 × layers × 2 kv_heads × 32 × L × batch_size   (fp32 bytes)
```

`ModelConfig.kv_cache_bytes(seq_len)` computes the KV figure if you later add an
incremental cache. Today the whole prefix is recomputed per call, which is fine
at these sizes.

Next: [Calibration →](08-calibration.md)
