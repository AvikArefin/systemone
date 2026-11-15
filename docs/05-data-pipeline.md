# 05 — Data pipeline

How a human-readable example becomes a batch of tensors, and why the storage
layout is shaped the way it is. File: [`data.py`](../src/systemone/data.py).

## Step 1 — the `Example`

Everything starts from one frozen dataclass:

```python
Example(
    state="I was charged twice for the same order.",
    question=Choice(instructions="Which department?", criteria=["billing", "security", "support"]),
    label="billing",
    distribution=(0.8, 0.15, 0.05),   # optional soft target
)
```

`__post_init__` validates aggressively: the label must be one of the criteria, a
score label must be a valid level index, a distribution must have the right
length, be non-negative, and sum to 1. Bad data fails at construction, not three
hours into training.

The optional `distribution` is what makes soft labels work — see
[Training](06-training.md).

## Step 2 — one example becomes several rows

```python
rows, target, kind, distribution = rows_for(example, tokenizer)
```

| Question | Rows produced | `target` |
|---|---|---|
| `noul` | 1 | `1.0` if true else `0.0` |
| `choice` | one per option | index of the correct option |
| `score` | one per level | index of the correct level |

This expansion is the whole reason a "batch of 32" can mean 179 rows.

## Step 3 — the flat token cache (`EncodedSet`)

Naively you would store `list[list[int]]` per example and re-tokenise every
epoch. That is what this project originally did, and it was slow: tokenising,
then rebuilding padded tensors row by row in a Python loop.

The replacement stores **no padding at all**, just two flat buffers plus offsets:

```
state_data    [tok0 tok1 tok2 tok3 tok4 tok5 tok6 tok7 tok8 ...]      int32
state_offsets [0        3        7        9 ...]                     int64
               │        │        │
               │        │        └─ example 2 = data[7:9]
               │        └────────── example 1 = data[3:7]
               └─────────────────── example 0 = data[0:3]
```

Three offset arrays, one per level of nesting:

| Array | Indexed by | Gives |
|---|---|---|
| `state_offsets` | example | slice of `state_data` |
| `row_offsets` | example | which global rows belong to it |
| `query_offsets` | row | slice of `query_data` |

Plus the per-example targets, kinds, and distributions. This is the same idea as
CSR sparse matrices: one flat array, and offsets to slice it. It is sometimes
called a **jagged** or **ragged** layout.

### Why it is faster and smaller

```python
# old: per-row Python loop, ~179 tensor creations per batch
for row, seq in enumerate(sequences):
    ids[row, :len(seq)] = torch.tensor(seq)

# new: slices are views; padding happens once, in C++
states  = [state_data[s:e] for s, e in ...]           # no copy
query_ids = pad_sequence(queries, batch_first=True)   # one C++ call
```

Measured on this machine, per batch of 32:

```
tokenise + build rows   1.78 ms
cached collate          0.53 ms      (3.4x faster)
```

End-to-end the same 6-epoch run went **394 s → 296 s (1.33×)**. Note the honest
scope: tokenising was *not* the expensive part; the per-row tensor loop was.

Storage for a 26k-example cache: ~1 MB. For 285k examples: ~274 MB
(`32.7M + 35.7M` int32 tokens).

## Step 4 — collate one batch

`EncodedSet.collate(indices)`:

1. Slice the states for the chosen examples.
2. Expand each example's rows via `row_offsets`, slice their queries.
3. `pad_sequence` both to the **widest row in this batch** — not the global max.
4. Build `group_ids` with `repeat_interleave(arange(B), rows_per_example)`.
5. Broadcast the state rows up to row count with `index_select`.

Masks are derived, not stored: `mask = ids != pad_id`. Valid because a real token
can never be ID 0.

Returned `Batch` fields: `state_ids`, `state_mask`, `query_ids`, `query_mask`,
`group_ids` (all int32/bool tensors), plus `targets`, `kinds`, `distributions`
(plain Python lists, indexed by group).

## Step 5 — caching to disk, safely

```python
dataset = load_or_build(path, examples, tokenizer, cfg)
```

- If the file exists **and** matches the current tokenizer and config, load it
  (5 ms for 26k examples).
- Otherwise rebuild and save.

Matching uses a fingerprint: SHA-256 of `tokenizer.model`, truncated to 16 hex
chars. This closes a nasty failure mode — retraining the tokenizer with the same
`vocab_size` produces different IDs, and a vocabulary-size check alone would
silently reuse the wrong cache.

## Gotchas worth knowing

- **Empty sequences raise.** A state that tokenises to nothing, or a candidate
  that truncates to nothing, is a data bug, not something to pad.
- **A candidate must never be truncated away.** `EncodedSet.build` raises if a
  `choice`/`score` row has no separator, because a group whose rows all collapse
  to one encoding cannot be answered. This guard exists because front-only
  truncation silently deleted the candidate from 44.8% of S1MB rows; see
  [Tokenizer and prompts](04-tokenizer-and-prompts.md) for the budget rule.
- **`max_candidates=8` is a training default, not a limit.** Training subsamples
  large option sets to 8 (always keeping the gold one) and shuffles order, so the
  model cannot memorise position. Benchmark scoring must *not* do this — it needs
  the original candidate set and order, so `load_s1mb(..., max_candidates=None,
  shuffle_options=False)`. The schema accepts up to 256 options, because S1MB has
  choices with 151.
- **Drops are reported.** `load_typed_decisions` counts any decision it cannot
  parse and prints a warning; pass `strict=True` to raise instead. Silently
  dropping 10.8% of the benchmark was how 9 subsets disappeared unnoticed.
- **Order matters.** `kinds`, `targets`, and `distributions` are parallel to the
  example indices, which is why the scoring code can zip a separate metadata list
  alongside the cache. `EncodedSet` also stores `benchmarks`, `candidate_ids`, and
  `values` so scoring can group by benchmark without re-parsing the dataset.
- **The cache is versioned.** `CACHE_VERSION` invalidates caches built by older
  code. Without it, a cache containing deleted candidates would be silently
  reused, because the fingerprint only covers the tokenizer.

Next: [Training →](06-training.md)
