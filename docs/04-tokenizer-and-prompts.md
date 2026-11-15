# 04 — Tokenizer and prompts

The network only understands integers. This page explains how text becomes token
IDs, and exactly how a question is laid out inside one row.

Files: [`tokenizer.py`](../src/systemone/tokenizer.py),
[`bpe.py`](../src/systemone/bpe.py), [`prompt.py`](../src/systemone/prompt.py).

## Two tokenizers, one interface

Both expose `encode(text) -> list[int]`, `decode(ids) -> str`, `vocab_size`,
`pad_id`, `sep_id`.

### `ByteTokenizer` — the baseline

One token per UTF-8 byte, offset by 2. A 300-character string is 300 tokens (more
for non-ASCII). Vocabulary is fixed at 258 and needs no training.

Useful for tests and for seeing how much subword tokenisation buys you. The real
runs use the other one.

### `SentencePieceTokenizer` — what training uses

A BPE model trained on a corpus. Multi-character pieces become one token, so a
sentence costs ~3-4× fewer tokens.

```python
tokenizer = train_tokenizer(texts, "artifacts/<name>/tokenizer", vocab_size=2048)
```

`train_tokenizer` writes a corpus file, runs `SentencePieceTrainer` in BPE mode
with `character_coverage=1.0`, deletes the corpus, and returns the wrapper.

Fixed IDs shared by both tokenizers:

| ID | `ByteTokenizer` | `SentencePieceTokenizer` |
|---|---|---|
| 0 | `[PAD]` | `[PAD]` |
| 1 | `[SEP]` (`␟`, U+241F) | `<unk>` |
| 2 | — | `[SEP]` (`␟`, U+241F) |

`sentencepiece` reserves `unk_id=1`, so the separator lands at ID 2 in a trained
tokenizer. Always read `tokenizer.sep_id`; never hard-code 1.

`␟` is declared as a `user_defined_symbol` so the trainer reserves it as an
atomic piece instead of splitting it.

## How text is cut to size

A row is `instructions + ␟ + candidate`, and the two halves do **not** share a
budget equally. The candidate is the only part that differs between the rows of a
group, so a row that loses its candidate cannot be answered. `prompt._split_budget`
therefore protects a floor for the candidate and gives the rest to the
instructions, which are identical across the group and cost only context:

```python
available   = limit - 1                    # one token for the separator
cand_budget = min(len(candidate), max(1, available - MIN_INSTRUCTION_TOKENS))
instr_budget = available - cand_budget
```

Both halves then use `truncate_head_tail`, which keeps 75% of the beginning and
25% of the end. For a state that keeps both the framing ("Customer note: …") and
the conclusion ("… asks us to delete the records"), which is usually where the
decision signal lives:

```python
def truncate_head_tail(ids, limit):
    if len(ids) <= limit: return ids
    head = limit * 3 // 4
    return ids[:head] + ids[-(limit - head):]
```

Limits: 192 tokens for a state, 64 for a query in the deployed config. The query
limit was 48 until the encoding fix. Measured on the full S1MB test split
(14,009 cases, 26,269 decisions, every candidate kept, `scripts/measure_budget.py`):

```
budget  kind      rows       cut  instruction kept  candidate kept  whole rows
  64  choice  119,497      1.8%             99.3%           99.8%       98.2%
  96  choice  119,497      1.0%             99.8%          100.0%       99.0%
 128  choice  119,497      0.2%            100.0%          100.0%       99.8%
 192  choice  119,497      0.0%            100.0%          100.0%      100.0%
  64  score    25,190     88.8%             25.4%          100.0%       11.2%
  96  score    25,190     87.8%             36.0%          100.0%       12.2%
 128  score    25,190     87.6%             46.1%          100.0%       12.4%
 192  score    25,190     87.2%             66.1%          100.0%       12.8%
  64  noul     12,357     61.4%             61.5%               —       38.6%
  96  noul     12,357     49.1%             69.3%               —       50.9%
 128  noul     12,357     46.0%             75.0%               —       54.0%
 192  noul     12,357     42.0%             84.4%               —       58.0%
 384  all  —                    0.0%            100.0%          100.0%      100.0%
```

**The smallest budget that keeps every row whole is 384.** At 128, `choice` is
effectively solved (99.8%) but `score` is not: its instruction is a ~250-token
rubric, so 128 keeps 46% of it and only 12.4% of rows survive whole. `noul` sits
at 54%. If the goal is "99% or better for every kind", 384 is the number; 128 is
the number for `choice` alone. Row lengths: `choice` p50 28 / p90 44 / p99 91,
`score` p50 297 / p90 312 / p99 324, `score` max 324.

This table is a *coverage* measurement, not a recommendation for the current
checkpoint. Scoring the same rows at a larger budget makes the model worse, not
better, because it was trained at 192/64 — see
[Experiments](11-experiments.md#raising-the-budget-at-inference-makes-it-worse).

Two corrections to what this page said before. First, 64 is **not** the point
where `choice` stops being cut: 1.8% of rows are still over budget at 64 and
100% survive only at 192. An earlier table claiming 100% was not reproducible and
is superseded by the numbers above.

Second, `score` was missing from that table and it is the worst kind. Most score
rows come from the relevance subsets (NanoBEIR, nanocoir), whose instruction is a
~250-token rubric; at 64 it keeps 25.4% of the instruction and only 11.2% of rows
survive whole. The handful of score subsets with short instructions (10-58 tokens)
are unaffected, but they are a minority of the rows. `noul` is next, because its
instruction carries a full policy definition. Neither is fixed by 64 — `score` is
still only 26.2% intact at 256 — so both are known costs, not something this
budget addresses. The 384 row is the first budget that does fix them.


`EncodedSet.build` raises if any `choice`/`score` row has no separator, because a
group whose rows all collapsed to one encoding is unanswerable. Before this check
existed, front-only truncation silently deleted the entire candidate in ~45% of
S1MB rows.

## The prompt format

One row is rendered with three parts, in this order:

```
[instructions text]  ␟  [candidate text]
```

and then the state is prepended by the batch layout, not by the text:

```
row = state ␟ instructions ␟ candidate
      └────── known before ──────┘
```

In code:

```python
def _render(tokenizer, instructions, candidate):
    ids = tokenizer.encode(instructions)
    if candidate is not None:
        ids = ids + [tokenizer.sep_id] + tokenizer.encode(candidate)
    return ids
```

- `noul_query` → instructions only.
- `choice_query` → instructions + `␟` + that option.
- `score_query` → instructions + `␟` + that level.

Because each row carries its own full instructions, a batch is just a set of
independent rows. Nothing needs to be shared structurally — the sharing is an
optimisation (the same state vector is reused, see
[Data pipeline](05-data-pipeline.md)).

## Why the instructions repeat per candidate

It wastes tokens: a 4-option question repeats the instruction 4 times. The
alternative — one row holding instructions and all candidates — would require the
mask to prevent candidates from seeing each other, and the head to pool each
candidate separately.

Repetition keeps the code simple and the batching uniform. An earlier version of
this page claimed "the 48-token query cap usually leaves room" — that was false,
and it was the assumption the truncation bug hid behind. Measured on S1MB:
instructions run to p50 27, p90 599, max 617 tokens, while the budget was 48. Half
the noul queries sat exactly at that cap, and 21.6% of candidate rows had **no room
left for the candidate at all** before the split gave the candidate priority. The
cap does not usually leave room; it usually binds.

The real cost is paid in the token cache ([Data pipeline](05-data-pipeline.md)):
query tokens are ~53% of everything stored.

## What the model actually sees

For a `choice` row with the deployed config:

```
positions 0..191   state tokens, truncated head/tail, padded to 192
positions 192..255 instructions ␟ option text, at most 64 tokens in total,
                   split with a floor for each half, padded to the batch width
```

The 64 tokens are shared between the instruction and the option, not given to the
instruction first. Padding is per batch, not per dataset — a batch of short rows
wastes less.

Next: [Data pipeline →](05-data-pipeline.md)
