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
hotchpotch/s1mb-dataset · nonexistent · test is not in the local cache
```

Nothing is fetched to fix that. The startup line prints how many configs are
cached per repo:

```
local cache only: {'hotchpotch/s1mb-dataset': 106, 'hotchpotch/bekko-...': 179}
```

## Choosing what to look at

The header has a **repository** selector and a **split** selector. Both repos are
always available, and the split list follows the config you pick: the benchmark
repo only ships `test`, while the training repo ships `train`, `validation` and
`test` depending on the config.

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

Every row is expandable to show the tokens it actually gets. Rows are budgeted
separately from the document, and the candidate is protected: the instructions
absorb the truncation, so a long option is never the thing that gets cut. A row
whose instructions are 357 tokens and whose budget is 64 says so plainly.

## The vocabulary sheet

The **vocab** button (or `v`) opens every piece in the tokenizer with its id, what
it decodes to, and how often the current config uses it. Counts come from the
first 400 cases of that config: state text, instructions, and option text.

This is the view for tokenizer questions. On `arc` training data:

```
873 of 2,048 pieces used · 42,509 tokens over 400 of 3,352 cases
1,175 pieces never appear
```

More than half the vocabulary never fires on a single config, which is the
counter-argument to any claim that the embedding table is doing useful work.
Pieces with zero uses are printed in red, the table sorts by count or by id, and
the filter takes a word or a single character.

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
  `#repo=s1mb-dataset&split=test&config=arc&case=arc:ARC-Challenge:test:27&tab=dec`
  can be pasted into a bug report. Add `vocab=1` to open the sheet.
- The token view needs a tokenizer. It uses `artifacts/bekko_tiny/tokenizer.model`
  unless you pass `--tokenizer`. Without one the browser still works and says so.
