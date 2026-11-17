# Data browser

A local page for walking through the datasets by hand. It reads the HuggingFace
cache directly, so nothing is downloaded and nothing is preloaded.

```
uv run python scripts/browse.py
uv run python scripts/browse.py --port 8765 --state 384 --query 128
```

Then open <http://127.0.0.1:8765>.

## Offline, by construction

Two separate guarantees, because the first one is not enough on its own.

The script sets `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` and
`TRANSFORMERS_OFFLINE` before anything imports `datasets`, so even a stray code
path cannot reach the Hub. And it does not call `load_dataset` at all. It finds
the arrow file for the config and split on disk and opens it with
`Dataset.from_file`, so the only thing that can happen is reading a local file.

A config that was never downloaded is therefore reported as missing:

```
hotchpotch/bekko-system-one-dataset-v0 · nonexistent · test is not in the local cache
```

Nothing is fetched to fix that. The startup line prints how many configs are
cached:

```
local cache only: {'hotchpotch/bekko-system-one-dataset-v0': 182}
```

## Choosing what to look at

The header has a **repository** selector, a **split** selector, and a
**tokenizer** selector. Everything is read from bekko, and the split list follows
the config you pick: train-only configs show `train`, the 106 benchmark configs
show `test` (many also have `train` and `validation`).

### The tokenizer selector

Every `tokenizer.model` found under `artifacts/` is offered, labelled by its
directory, with its vocab size and how many foreign single-character pieces it
holds. A comparison version also shows the description from its `meta.json`:

```
bekko_tiny · 2,048 pieces · 640 foreign
tokenizer_versions/v0_raw · 2,048 pieces · 735 foreign · no filtering
tokenizer_versions/v1_strip · 2,048 pieces · script strip only
tokenizer_versions/v3_floor10 · 2,048 pieces · strip + fold + floor 10
tokenizer_versions/v4_dedup · 3,072 pieces · dedup + split digits + Greek cleanup (new default)
```

Switching rebuilds the token views and re-counts the vocabulary sheet for the new
model, so the same config can be compared under two tokenizers without a restart.
`--tokenizer` only picks which one is selected at startup; the rest stay reachable
from the page. A `SentencePieceProcessor` lives only as long as the selection, so
choosing between models does not accumulate them.

This is how the corpus-filtering work is verified by eye: open `codesearch` and
switch from `v0_raw` to `v3_floor10`. The foreign count goes to 0 and the token
count for the same 400 cases drops. Switch again to `v4_dedup` to see the corpus
change shape: `doctor`, `legal` and `friend` become whole pieces, numbers stop
merging (`▁2027` is gone), and rubric words fragment (`▁instructions` becomes two
pieces, `▁probabilities` four).

Three panes follow: **configs** → **cases** → **one case**.

- **Configs** are the dataset families in the cache, filterable by name.
- **Cases** list case ids with decision count, kind badges, and document length.
  The first box filters by case id; the second **jumps** to one: type part of an
  id, press enter, and that case opens with its neighbours in the list.
- **One case** has three tabs: the document, its decisions, and the raw row.

## The document tab

Two views. **Text** is the document as a reader sees it. **Tokens** is what the
model sees: the tokenizer's pieces in order, with the part beyond the budget
shown as a red `⋯ N tokens dropped ⋯` marker.

The budget in the header is live. Change it, press apply, and the marker moves.
That is the point of this tab: the difference between "1,051 tokens" and "the
model sees 192 of them" is the single most misleading thing about this project's
numbers, and it is now one click to see.

Truncation keeps the head and the tail (three quarters from the front, one
quarter from the back), so the marker sits in the middle of the document, not at
the end.

## The decisions tab

One card per decision. Each shows the instructions exactly as the model receives
them — including the label definitions for `noul`, which the dataset supplies and
which are easy to lose — the options with the gold one highlighted, the target
probabilities as bars, and the instruction token count.

`ranking` decisions look different because they have no `type` and no `criteria`:
their candidates live in `documents`. The card lists those documents as options,
marks the positive one, and shows the `scoring` mode. The browser reads them
straight from the row because the training parser drops them — `arxiv` and
`agnews` are one ranking decision per case, so hiding them as "unsupported" hid
the whole config. Nothing here changes training; the model still has no ranking
head.

The **show tokens** button, or `t`, is the same control as on the document tab.
It swaps the whole tab to the tokens each row actually gets — one group per
option, or the single row for `noul` — labelled with its token count and any
truncation. Press it again to return to instructions and options. Rows are
budgeted separately from the document, and the candidate is protected: the
instructions absorb the truncation, so a long option is never the thing that
gets cut. A row whose instructions are 357 tokens and whose budget is 64 says so
plainly: the label reads `64/357`, and the token strip ends with a red
`⋯ 293 tokens dropped ⋯` marker so the missed count sits with the tokens.
Ranking decisions have no query tokens, so they stay in text form.

## The vocabulary sheet

The **vocab** button (or `v`) lists the whole vocabulary: one row per piece the
tokenizer knows, which is also one row of the model's embedding table. Each row
shows its number, the raw piece, what it decodes to, how often it was seen, and
its share of the tokens counted.

The sheet says all of this in its own words, because the three numbers are easy to
misread:

- **All 2,048 pieces are listed**, used or not. This is the model's actual
  embedding table, not a sample of it.
- **"times seen"** is a count inside *this config only* — `arc`, split `train`. It
  is not the dataset as a whole and not one example.
- **Counts are a sample**: the first 400 of that config's cases, covering state
  text, instructions, and options. The cap exists so the sheet opens quickly.
- **"% of tokens"** is that piece's share of the counted tokens, so the column adds
  up to 100%.

On `arc` training data:

```
873 of 2,048 pieces used · 1,175 never seen
42,509 tokens over 400 of 3,352 cases
```

"Only never seen" filters to the dead rows. On `arc` they include `<pad>`, `<unk>`,
stray braces and quote fragments, and near-duplicate merges like `intent` next to
`intent`. A row the data never reaches is a row that never gets a useful gradient,
which is the concrete form of "26% of a 1M model is embedding".

"Only foreign" narrows to the pieces that are a single character from a script the
corpus should not contain — the ones `train_tokenizer` strips. They are dead weight
by construction: no English row can spend them. The sheet shows how many exist and
the count per piece, which is what makes the case for filtering the corpus rather
than shrinking the vocabulary.

## The foreign sheet

The **foreign** button scans every cached config in the selected repo and reports
which ones contain a character this project does not train on — Chinese, Cyrillic,
Arabic, Hebrew, Thai, emoji — grouped by script.

It exists because "which dataset uses the Cyrillic letters" is not answerable by
opening configs one at a time. On the training repo:

```
synthetic_relevance__nanocoir           8 / 250 rows   8,048 chars   cyrillic 8,048
synthetic_relevance__codesearch        11 / 250 rows     500 chars   cyrillic 329 · cjk 115 · arabic 56
synthetic_relevance__bge_m3__miracl_en  1 / 250 rows     200 chars   cyrillic 200
qasper                                 18 / 250 rows   1,256 chars   cjk 873 · cyrillic 150 · arabic 115
```

Read the shape, not just the size: `nanocoir` carries 8,048 Cyrillic characters on
**8 of 250 rows**, so it is a few heavily non-English documents rather than a
pervasive problem. `qasper` spreads its foreign text across 18 rows. The filter box
matches a config name or a script, so typing `cyrillic` narrows the table. Each
config is sampled at its first 250 cases — enough to find what a config contains,
not to measure its exact volume. Every split is scannable, and the Cyrillic text
in the test configs comes from the same `nanocoir` family.

## The raw tab

The source row, untouched: `state_json`, every decision, every target. When a
value looks wrong in the parsed view, this is the place to check whether the bug
is ours or the dataset's.

## Keyboard

`j` / `k` next and previous case, `1` `2` `3` switch tabs, `t` toggle tokens,
`/` filter cases, `v` vocabulary, `escape` close the sheet.

## Notes

- Cases are parsed on demand, one at a time. A family with 300k decisions is fine
  to open; only the case you click is turned into typed decisions.
- Every page is addressable:
  `#repo=bekko-system-one-dataset-v0&split=test&config=arc&case=arc:ARC-Challenge:test:27&tab=dec`
  can be pasted into a bug report. Add `vocab=1` to open the sheet.
- The token view needs a tokenizer. It starts on
  `artifacts/bekko_tiny/tokenizer.model` unless you pass `--tokenizer`; every other
  model under `artifacts/` is selectable from the header. Without any model the
  browser still works and says so.
