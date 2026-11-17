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
tokenizer = train_tokenizer(texts, "artifacts/<name>/tokenizer", vocab_size=3072)
```

`train_tokenizer` writes a corpus file, runs `SentencePieceTrainer` in BPE mode
with `character_coverage=1.0`, deletes the corpus, and returns the wrapper.

**Fixed: the corpus is filtered before training.** `character_coverage=1.0`
reserves a vocabulary slot for every distinct character in the corpus, however
rare, which is the right default for a mixed-language model and the wrong one
here. The bekko training corpus has 1565 distinct characters; the top ~190 cover
99.99% of the text. The rest is noise that still costs a slot — stray CJK from
`open_jev__phone-extraction-control-v1` (`Memo: café / 联系卡`), Chinese song
titles in `bge-hn-hotpotqa`, Korean and Japanese article titles in `dbpedia` and
`fineweb`, manga titles in `boolq`, plus emoji (`✅`, `👍`, `😭`) and the invisible
`\u200d` joiner and `\ufe0f` variation selector.

`train_tokenizer` now runs `strip_foreign` over the corpus, which drops
characters from `FOREIGN_SCRIPTS` and from `EMOJI_RANGES`. Coverage stays at 1.0.
Measured over 148 cached bekko configs (51,619 cases, 552,862 texts):

| | raw corpus | stripped corpus |
|---|---|---|
| single-char pieces | 1,170 | 451 |
| of those, foreign or emoji | 727 | **0** |
| tokens per character | 0.3680 | 0.3208 |

That is 719 freed embedding rows and 12.8% fewer tokens, with no `<unk>` on
held-out English text.

**What the strip does and does not touch.** It edits the *tokenizer's* copy of the
text, nothing else. **No training or eval row is dropped** — the filter runs on
the corpus file written for `SentencePieceTrainer`, so `load_bekko_train` and
`load_s1mb` are unaffected and every case still trains and evaluates. Measured on
44,128 sampled training rows and 10,566 S1MB eval rows:

| | training | eval |
|---|---|---|
| rows dropped | 0 | 0 |
| rows containing a foreign char or emoji | 1.56% | ~3.2% |
| ...of which CJK/kana/Hangul | 1.15% | — |
| ...of which emoji/symbol | 0.11% | — |
| rows that now emit an `<unk>` | 1.61% | — |
| `<unk>` tokens as share of all tokens | 0.011% | 0.03% |
| tokens | −12.21% | −12.75% |

The catch is that a stripped character becomes `<unk>` rather than being cleanly
removed: a run of foreign text collapses to a single `<unk>`, which is why the
token count falls and the unknown rate rises together. That is the intended
trade — a `<unk>` on a disclaimer nobody can use costs one row of embedding,
where the individual characters cost 700.

Two things this does *not* break. Choice groups whose options collapse to
identical encodings: 6 exist on S1MB and **all 6 predate this change** — they are
duplicated option text in `hh_rlhf`, not a tokenizer artifact. The strip
introduces none. And `aqua_rat`, whose decisions carry `√`, `≤`, `π`: those stay
real tokens.

**Second pass: accents fold unconditionally, then rare characters are policed.**
Script filtering cannot express "rare", so it kept every Latin accent and every
Greek letter. Two separate rules were needed, and getting them in the right order
mattered.

*Rule 1 — folding has nothing to do with frequency.* `é` and `e` are the same
letter. Whether one appears 68,841 times or twice says nothing about whether it
merits its own vocabulary slot. The first version of this gated folding on the
frequency floor, which left **298 accented letters** — `ý` (812 uses), `é`
(68,841), `ü` (59,766), `ö` (52,928) — sitting in the vocabulary beside their
ASCII bases: two slots and two embedding rows for one letter. Folding now runs
first and unconditionally.

*Rule 1a — compose before folding.* Scraped and email text arrives decomposed:
`e` + U+0301 rather than `é`. Folding cannot see that pair — `e` is ASCII and
U+0301 has no base letter — so it passed through, and SentencePiece's own NFKC
pass recomposed it into `é`, putting the accented letter straight back into the
vocabulary. `prune_rare` now normalises to **NFC** first, so folding sees a real
`é`. NFC rather than NFKC: NFKC would also expand `ﬁ` → `fi` and `⅓` → `1⁄3`,
changing text the filter has no business touching. With this, the cleaned corpus
yields 61 non-ASCII letters after NFKC instead of 121, and none of them are
accented Latin.

*Rule 2 — whatever is still non-ASCII and rare is resolved, not ignored.* The
second pass only sees characters that folding could not remove, and gives each one
an explicit outcome:

| case | example | outcome |
|---|---|---|
| invisible character | `\u200c`, `\u0304` | dropped — renders as nothing |
| currency | `¥`, `₱` | **kept** — money must stay readable |
| maths notation | `∑`, `∩`, `←`, `√` | **kept** — no alternative spelling |
| anything else | `■`, `【`, `⅓`, `ἀ`, `ɜ` | **dropped** |

The early version had no fourth row: a symbol with no base letter fell through
untouched, so `■`, `【`, `《`, `⅓` and bare Greek (`ἀ`, `ὀ`, `ὁ`) kept slots
forever at any threshold.

**Folding is safe, measured.** The risk is not token count but collisions — two
distinct strings becoming identical. Across 44,065 choice groups in the training
corpus and 20,484 in S1MB, folding merges **zero** options. The merges that do
happen are exactly the intended ones: `León`/`Léon` → `Leon`, `für`/`fur`,
`Coöperatieve` → `Cooperatieve`, `Frédéric` → `Frederic`.

Threshold 10 then applies to what is left. The band between 3 and 9 holds 85
characters and almost none are English — `ě`, `ą`, `ṃ`, `ɜ`, `ʲ`, stray Greek.
Dropping them costs nothing measurable.

**A new vocabulary invalidates the weights, and now says so.** Retraining the
tokenizer leaves every checkpoint trained against the old one meaningless, and
`load_state_dict` cannot tell — it compares tensor shapes, and the same number
of rows is the same shape whichever pieces they represent. `save_checkpoint` writes a
`tokenizer_fingerprint` into `model.pt` and `train()` stamps one into
`training_state.pt`; both are checked on load, so a swapped vocabulary raises
`VocabMismatch` instead of training on silently relabelled tokens. Details in the
[data pipeline](05-data-pipeline.md#the-same-fingerprint-guards-the-weights).

## The vocabulary was shaped by a corpus that repeated itself

An `aegis2` document encoded as:

```
·do|ct|or   ·le|g|al   ·c|ann|ab|is   ·ha|v|ing   ·f|ri|end   ·n|at|ur|ally
```

`doctor` in three pieces, `legal` in three, `cannabis` in four. Meanwhile the
vocab held whole pieces for words that only appear in rubrics:

```
·instructions  ·identification  ·interpretations  ·probabilities  ·contradiction
```

Both facts had the same cause: **the training corpus was not a sample of the
text the model reads.**

`collect_texts` emitted one string per *typed example*, and a choice case with
eight options becomes eight examples sharing one state and one instruction, so
the same document was fed to BPE eight times. Measured over 148 cached configs:

| | count |
|---|---|
| texts emitted | 552,862 |
| distinct texts | 71,917 |
| duplicates | 480,945 (**87.0%**) |
| distinct instruction strings | 3,398 |
| instruction emissions | 118,223 (97.1% redundant) |

BPE's merge budget is driven by frequency, so it went to whatever repeated
hardest. A single score-level description (`"4.0: A direct and sufficient answer
for the intended subject"`) was fed in **7,600 times**. Rubric vocabulary was
therefore cheap to merge into whole pieces, while `doctor`, which appears in
documents, was not. By characters the corpus was 74.7% state text, 18.1%
instructions and 7.2% options; by frequency the instructions were dramatically
over-represented relative to how varied they are.

**Fixed: `collect_texts` keeps one copy of each text.** The change is one line,
`list(dict.fromkeys(texts))`, keeping the first occurrence so the corpus order
stays deterministic. `deduplicate=False` remains only to rebuild the historical
comparison tokenizers.

**Dedupe alone gets partway.** Measured on the same corpus, same hyperparameters
and same character filters by
[`scripts/build_tokenizers.py`](../scripts/build_tokenizers.py):

| corpus | vocab | `doctor` | common words whole | document tokens | single-char pieces | tokens/char, repeated sample | tokens/char, distinct |
|---|---|---|---|---|---|---|---|
| repeated (`v3_floor10`, old) | 2,048 | 3 | 7/22 | 46 | 283 | 0.3230 | 0.3597 |
| deduplicated, before pruning | 2,048 | 2 | 9/22 | 40 | 278 | 0.3392 | 0.3515 |
| deduplicated + pruned (`v4_dedup`, new) | 3,072 | **1** | **17/22** | **36** | **260** | **0.3161** | **0.3288** |

The document is the fixed aegis2-style sentence
`prompt: I don't think my doctor would know about them. ...` (144 characters),
not a rubric string. Deduping removes 87% of the corpus and shortens that
document by 13%. `doctor` goes from three pieces to two, `legal` and `having`
from three to two, and `friend` from three to one. The rubric words move the
other way on purpose: `instructions` 1→2, `identification` 1→3, `probabilities`
1→4. The middle row is the deduplicated corpus before the second round below; it
is a measurement, not a shipped artifact.

**Second round: digits and Greek.** Two kinds of slot were still being paid for.
Dates in the voucher/policy rows made `▁2027` and `▁2027-0` real pieces, while
`doctor` occurs 973 times in the deduplicated corpus and the 2,048th merge sits
around frequency 1,000 — it missed by a hair. `train_tokenizer` now passes
`split_digits=True`, so every digit is its own piece and no number can merge;
60 multi-digit pieces disappear. Accented Greek vowels now fold to their Latin
lookalike (`ό` is a Greek omicron with a tonos, so NFKD stops at Greek `ο`; the
fold table maps it to `o`), and Greek that is not a reserved maths symbol is
dropped however frequent. That removes 11 dead slots and leaves only
`π σ Δ α β μ λ`.

Pruning alone did not rescue `doctor`: the freed slots went to other frequent
merges. The binding limit was the budget. 3,072 slots hold `doctor`, `legal`,
`friend`, `anything` and `either` whole and take the document to 36 tokens; on
the 128-wide model the extra 1,024 rows cost 131k parameters (1.00M → 1.13M,
+13%). `cannabis` occurs 13 times and `naturally` 206, so they stay split at any
vocabulary this size. That is the right trade: a piece for a 13-use word spends
a slot a common word needs.

After both rounds the same document is:

```
·doctor ·legal ·c|ann|ab|is ·having ·friend ·n|at|ur|ally
```

**States-only was rejected.** Training only on the state text keeps more common
words whole, but it drops every instruction and option string, which the model
reads for `score` and `noul` rows, so it optimises for a distribution the model
does not see at inference. Dedup keeps every distinct text and removes only the
repetition.

Both tokens-per-char columns improve with the pruned 3,072 model: the repeated
sample goes from 0.3230 to 0.3161, and the distinct corpus from 0.3597 to
0.3288. The repeated column is a 1-in-20 sample of the 552,862-text corpus, so it
is not directly the full-corpus 0.314 in the character-filter table above; every
variant is measured on the same sample.

**Consequence: retrain, and rebuild the tokenizer.** A new vocabulary makes
every checkpoint trained against the old one meaningless. The fingerprint guard
already refuses a mismatch (`VocabMismatch`); see
[Data pipeline](05-data-pipeline.md#the-same-fingerprint-guards-the-weights).
`train_main.py` reuses an existing `tokenizer.model`, so building the new
vocabulary means deleting it or using a different checkpoint name.

Reproduce the table with `uv run python scripts/build_tokenizers.py`. It reads
the cached configs once, pickles the corpus and builds each variant, writing
`artifacts/tokenizer_versions/<name>/meta.json`; `v4_dedup` sits beside the four
older models in the data browser. The corpus is capped at 400 cases per config,
the same cap the earlier measurements used, so the numbers are comparable; the
cap is a measurement choice, not a training one.

**Where the 2,048 slots went** (measured on `v3_floor10`):

| | count |
|---|---|
| single-character pieces | 283 |
| ...of which non-ASCII | 116 |
| single-character pieces that are ASCII | 167 |
| extra slots held by case variants (e.g. `▁the`, `▁The`, `▁THE`) | 153 |

The new `v4_dedup` has 3,072 slots and 260 single-character pieces, 88 of them
non-ASCII, against 283/116 here. The breakdown below is still the old 2,048
model; the case-variant slots remain a real cost.

The 167 ASCII single-character pieces are unavoidable — English needs its
alphabet. The 116 non-ASCII in the old 2,048 model broke down as:

| what | count | examples |
|---|---|---|
| reserved currency and maths | 30 | `£ ¥ € ∑ √ ∞ ≤ π σ Δ` |
| typography and symbols | 27 | `’ “ ” — – © ® § °` |
| IPA / phonetic | 24 | `ˈ ɛ ɪ ʃ ə ɹ ʒ` |
| Greek | 16 | `ι κ τ ς ε γ ο ρ` |
| accented or special Latin | 12 | `ı ł æ ð ø Þ ŋ ß Ø Đ` |
| invisible / combining | 6 | `\u0303 \u0301 \u0308 \u0323` |
| other | 1 | `ー` |

Only the top row is deliberate. The IPA is pronunciation notation from
`helpsteer2`; the 16 Greek letters were mostly maths that never grew into a
merged piece. `v4_dedup` drops Greek that is not reserved maths and folds
accented Greek vowels, so only the reserved maths letters remain. The 12 accented
Latin have no base letter to fold to (`æ`, `ø`, `ß`, `Þ` are distinct letters,
not decorated `a`, `o`, `s`, `T`). The 6 combining marks survive the fold because
they sit above the floor and attached to characters NFC cannot compose. At 116
slots this is no longer the main cost — the 153 case-variant slots are — but it
is not "deliberate reservation" either, and the frequency floor cannot remove
what appears more than 10 times.

## Why building a tokenizer is slow

`SentencePieceTrainer.train` itself takes **1.3 seconds** on this corpus. The
build takes minutes because of what happens around it:

| stage | time |
|---|---|
| list cached configs | 4.6s |
| load datasets from HF cache and parse | **444s** |
| collect_texts | 0.03s |
| join into one string | 0.15s |
| clean_corpus (strip + fold + floor) | 26.8s |
| write corpus file (197 MB) | 0.17s |
| **SentencePieceTrainer.train** | **1.33s** |

The trainer is not slow. Loading 148 datasets from parquet and parsing 118,223
decisions into typed objects is 97% of the wall clock, and it happens *before*
`train_tokenizer` is called. If the corpus is already in memory, rebuilding the
BPE model is a two-second operation — so the fix for a slow rebuild is to cache
the corpus, not to touch the trainer.

One real inefficiency was in the fold pass: it called `fold_char` (which runs
`unicodedata.normalize`) once per character, 89s over 196M characters. It now
builds a translation table once from the corpus alphabet and uses
`str.translate`, which is **14.8× faster at 6.0s** and byte-identical.

Sweep, on the same corpus (552,862 texts, 196M characters, 1,565 distinct):

| threshold | chars kept | vocab single-char pieces |
|---|---|---|
| 1 (off) | 486 | 369 |
| 3 | 396 | 323 |
| 5 | 361 | 301 |
| **10** | **311** | **~262** |

**Symbols get a separate mechanism, because frequency is not enough.** `∑` and `∩`
both occur exactly twice, yet the final vocabulary cut kept `∩` and dropped `∑` —
BPE merges decide which candidates survive, and corpus text cannot control that.
`MATHS_SYMBOLS` and `CURRENCY_SYMBOLS` are therefore passed to SentencePiece as
`user_defined_symbols`, which guarantees a piece regardless of frequency.
`reserved_symbols()` narrows the list to symbols actually present, since reserving
an absent one spends a slot the data never reaches.

Measured over the full corpus, the filter changes 239 of 552,862 texts (0.04%) and
**no text becomes empty**.

Five versions are built for comparison under `artifacts/tokenizer_versions/`:

| version | what it is |
|---|---|
| `v0_raw` | no filtering |
| `v1_strip` | script strip only — the previous shipped behaviour |
| `v2_floor3` | strip + frequency floor 3 |
| `v3_floor10` | strip + unconditional fold + floor 10 — the repeated corpus, 2,048 pieces |
| `v4_dedup` | deduplicated corpus + split digits + Greek cleanup + strip + fold + floor 10 — the new default, 3,072 pieces |

Lowering `character_coverage` instead was the obvious fix and the wrong one.
It is a percentile, not a filter: at 0.9995 it also discarded `Z`, `Q`, `é`, `£`
and `–`, which English rows do use. Filtering only the foreign scripts removes
the same slots and keeps every character the corpus legitimately needs. The
filter deliberately keeps accented Latin, typography (`—`, curly quotes), and
math symbols (`→`, `≤`, `∞`, `√`) — `aqua_rat` and the relevance rows need those,
and they are not emoji despite sitting in the same Unicode neighbourhood.

Models for side-by-side comparison: five versions under
`artifacts/tokenizer_versions/` (see the table above). The first four were built
by an uncommitted scratch loop; `scripts/build_tokenizers.py` rebuilds `v4_dedup`
and measures any of the five.

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
