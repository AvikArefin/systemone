# Data browser

A local page for walking through the datasets by hand. It reads the HuggingFace
cache directly, so nothing is downloaded and nothing is preloaded.

```
uv run python scripts/browse.py                  # S1MB test split
uv run python scripts/browse.py --repo bekko     # training data
uv run python scripts/browse.py --port 8765 --state 384 --query 128
```

Then open <http://127.0.0.1:8765>.

## What it shows

Three panes: **configs** → **cases** → **one case**.

- **Configs** are the 148 dataset families in the cache. Clicking one loads it; the
  file is read once and kept in memory for the rest of the session.
- **Cases** list the case ids in that family with their decision count, kind
  badges, and document length. The search box filters by case id.
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
quarter from the back), so the marker always sits in the middle of the document,
not at the end.

## The decisions tab

One card per decision. Each shows the instructions exactly as the model receives
them — including the label definitions for `noul`, which the dataset supplies and
which are easy to lose — the options with the gold one highlighted, the target
probabilities as bars, and the instruction token count.

Every row is expandable to show the tokens it actually gets. Rows are budgeted
separately from the document, and the candidate is protected: the instructions
absorb the truncation, so a long option is never the thing that gets cut.

## The raw tab

The source row, untouched: `state_json`, every decision, every target. When a
value looks wrong in the parsed view, this is the place to check whether the bug
is ours or the dataset's.

## Keyboard

`j` / `k` next and previous case, `1` `2` `3` switch tabs, `t` toggle tokens,
`/` focus the filter.

## Notes

- Cases are parsed on demand, one at a time. A family with 300k decisions is fine
  to open; only the case you click is turned into typed decisions.
- Every page is addressable: `#config=arc&case=arc:ARC-Challenge:test:27&tab=dec`
  can be pasted into a bug report.
- The token view needs a tokenizer. It uses `artifacts/bekko_tiny/tokenizer.model`
  unless you pass `--tokenizer`. Without one the browser still works and says so.
