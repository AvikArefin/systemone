# Done: the tokenizer corpus is deduplicated and pruned

`collect_texts` keeps one copy of each distinct text. `train_tokenizer` also
splits digits and cleans up Greek, and a new vocabulary defaults to 3,072 pieces.
The reasoning and full measurements are in
[docs/04-tokenizer-and-prompts.md](docs/04-tokenizer-and-prompts.md#the-vocabulary-was-shaped-by-a-corpus-that-repeated-itself).

Short version, re-derived with `scripts/build_tokenizers.py` on the same corpus
(400 cases per config) and character filters:

| corpus | vocab | `doctor` | common words whole | document tokens | single-char pieces |
|---|---|---|---|---|---|
| repeated (old `v3_floor10`) | 2,048 | 3 | 7/22 | 46 | 283 |
| deduplicated, before pruning | 2,048 | 2 | 9/22 | 40 | 278 |
| deduplicated + pruned (`v4_dedup`) | 3,072 | 1 | 17/22 | 36 | 260 |

Why the second round:

- `▁2027-0` and 59 other multi-digit pieces were dates and IDs. `split_digits=True`
  makes every digit its own piece, so numbers cannot win merges.
- `ό` is a Greek omicron with a tonos; NFKD stops at Greek `ο`, not ASCII `o`, so
  the old fold skipped it. Accented Greek now folds to Latin, and non-reserved
  Greek is dropped; only the maths symbols `π σ Δ α β μ λ` remain.
- `doctor` occurs 973 times and the 2,048th merge sits near frequency 1,000, so
  pruning alone did not rescue it. 3,072 slots do, for 131k extra parameters
  (1.00M -> 1.13M).

`v4_dedup` lives beside the older models under `artifacts/tokenizer_versions/`;
`uv run python scripts/build_tokenizers.py` rebuilds and re-measures it. A new
vocabulary invalidates existing weights, so training must rebuild
`tokenizer.model` (delete the old one or use a new checkpoint name) rather than
reuse `bekko_tiny`.
