# 09 — Datasets and benchmark

Where the data comes from, what shape it is in, and the contamination trap.
Files: [`corpus.py`](../src/systemone/corpus.py),
[`decontamination.py`](../src/systemone/decontamination.py),
[`hf.py`](../src/systemone/hf.py).

## The format everyone shares

The test split and the bekko training rows use **the same schema**, which is
why one parser handles both:

```
input.state_json      the shared state, as JSON
input.decisions[]     typed questions: id, type (noul/choice/score),
                      instructions_json, criteria[], documents[]
targets[]             aligned distributions: decision_id, ids[], probabilities[]
split, case_id, group_id, language, input_hash
```

`probabilities` are soft: `[0.7, 0.3]` is a real target, not an error. Keeping
candidate order and soft targets intact is explicit in both dataset cards.

## Evaluation: the test split

The `test` split is the published System One benchmark (S1MB), republished
inside bekko. The source repo (`hotchpotch/S1MB-english-v1`, formerly
`hotchpotch/s1mb-dataset`) and bekko are kept in sync: all 106 test configs were
compared column-for-column, and rows, order and values are identical.

| | |
|---|---|
| Configs (subsets) | 106 |
| Cases | 14,009 |
| Judgments | 26,269 |
| Splits | `test` only |
| Benchmarks | 137 (59 noul, 57 choice, 21 score) |

The benchmark is **evaluation-only**. There is no train split to look for. If you
train on it, you are training on the benchmark.

`load_test()` reads `split="test"` and defaults to `test_configs()`, the 106
configs that actually carry a `test` split.

## Training: bekko corpus

`hotchpotch/bekko-system-one-dataset-v0` is the dataset behind the
bekko-system-one models, published in the same format.

| | |
|---|---|
| Train subsets | 153 |
| Train cases | 6,589,188 |
| Train judgments | 8,421,771 |
| Cap | 300,000 judgments per subset, complete cases |
| Decontamination | two train cases removed for exact test matches (2026-10-08) |

`load_train(max_cases=N)` reads `train_configs()` and caps rows per subset,
because a full pass is ~262k steps at batch 32 — about 27 hours on this machine.

## Licensing — read before shipping

The corpus does **not** relicense its sources. Each subset keeps its upstream
terms:

- permissive: CC BY 4.0, CC0, Apache-2.0, MIT (most subsets)
- non-commercial: MS MARCO, GooAQ, some NanoBEIR sources
- unknown: AG News, Enron Spam, SCICite

The bekko **model weights** are a separate matter: their card says the license
"remains to be finalized" and does not assign one. That is why the project uses
the *dataset* and not the model. Laya's weights **are** Apache-2.0 if a
permissively licensed teacher is ever needed.

## The contamination trap (important)

Our first "improvement" was training on the test split. That produced
0.417 → 0.551 and felt like progress. It was not comparable to anything:

- Published benchmark scores come from models that never trained on it.
- Our number came from a model that trained on 80% of it.
- Even after switching to the bekko corpus we still share **dataset families**
  with the evaluation — bekko's own authors disclose 77 of 106 subset names in
  common. Familiarity with a task family inflates scores without proving
  generalisation.

What we do now:

1. Train on `bekko-system-one-dataset-v0`, never on the test split.
2. Evaluate on the test split with rows held out (different rows, same families).
3. Say "same-family exposure" out loud whenever quoting a number.
4. Treat the six synthetic generalisation subsets as the honest signal, since
   they test instruction-following rather than family familiarity.

## The state budget vs. the real corpus

`max_state_tokens` is 192. Measured on the test split, one row per case,
with the tokenizer the deployed checkpoint actually uses (`bekko_tiny`):

```
states                          14,009
truncated                        6,382   (45.6%)
tokens p50 / p90 / p99        168 / 1,196 / 4,231
longest state                   28,401 tokens
tokens dropped by the 192 cap   4,893,744  (72.7% of all state tokens)
tokens dropped by a 768 cap     1,217,712  (18.1% of all state tokens)
```

Nearly half the benchmark is cut, and 72.7% of all state tokens never reach the
model. For document benchmarks (`qasper`, `contract_nli`, `followir_*`, the
NanoBEIR subsets) the answer is often in the middle, which head/tail truncation
does not keep.

**These counts depend on the tokenizer.** An earlier version of this table
reported 7,788 truncated (55.6%) and a 39,163-token maximum; that was measured
with an earlier comparison tokenizer, which is a different BPE vocabulary over
the same text. Compare budgets within one tokenizer, never across.

Bumping this budget is a cheap accuracy win that no run so far has tested — every
headline number in [Experiments](11-experiments.md) was produced at 192. The cost
is quadratic attention, so it is a real trade-off rather than a free fix, but it
should be measured before "capacity" is blamed. A 768-token state still only
leaves 18.1% of state tokens behind, versus 72.7% at 192.

## What survives if nothing is truncated

The obvious alternative to raising the budget is refusing to truncate: only keep
cases that fit. That is a much bigger cut than it sounds, because a case is lost
if *any* of its candidates overflows.

```
budget      cases        decisions     rows (candidates)
192/64      32.8%        35.6%         61.4%
384/128     48.0%        47.2%         74.7%
1024/384    87.0%        84.2%         92.9%
1536/384    94.7%        90.7%         95.6%
```

At the deployed budget, dropping everything that overflows would leave **a third
of the benchmark**, and it would not be a random third: the survivors are the
short documents, which are the easy ones. By kind at 192/64 — noul 34.1% of
decisions, choice 54.0%, **score 7.8%**.

That is the argument for truncating rather than filtering. Truncation keeps the
benchmark honest and complete at the cost of the model seeing 27% of each state;
filtering keeps every state whole at the cost of reporting on a biased third. The
numbers above are the price of each.

### Measured: fit-only does not flatter the model

The obvious worry about reporting on the subset that fits is that it will look
better than it is. A checkpoint was trained at 384/128 on fit-only data
(`SYSTEMONE_FIT_ONLY=1`, 85.0% of the training decisions) and then scored both
ways:

```
scope       decisions   noul   choice   score   Task Avg
full           26,269    2.52     0.06    0.71       1.09
fit-only       12,224    2.48     0.05    0.29       0.94
```

Fit-only came out *lower* — 0.94 against 1.09 — so the "survivors are the easy
documents" worry did not show up in practice. Filtering removes the long
documents, and on the short ones this model has no advantage to protect. `score`
is the only kind that moves much (0.71 to 0.29), which is composition rather than
difficulty.

Both are still ~1%, which is the point worth keeping: the subset you report on
does not rescue a model that has learned only the label marginals.

## Configs that cannot fit even at 1024/320

A larger budget does not rescue everything. Measured on the first three cases of
every cached config with `bekko_tiny`, **32 of 182 configs** still overflow a
**1024-token state** or a **320-token query**. Those rows cannot be trained
without cutting most of their text, so `corpus.UNFIT_CONFIGS` marks them and
`train_configs()` / `load_train()` leave them out by default.

State over 1024 (15):

| config | state tokens |
|---|---:|
| `qasper` | 26,977 |
| `contract_nli` | 5,511 |
| `helpsteer3` | 5,189 |
| `synthetic_relevance__trivia_qa` | 1,884 |
| `synthetic_relevance__nanocoir` | 1,715 |
| `open_jev__drone-control-v1` | 1,664 |
| `open_jev__amount-extraction-control-v1` | 1,502 |
| `open_jev__context-retention-control-v1` | 1,472 |
| `open_jev__browser-control-v1` | 1,415 |
| `synthetic_relevance__fineweb` | 1,408 (query 324 too) |
| `open_jev__email-selection-control-v1` | 1,381 |
| `open_jev__phone-extraction-control-v1` | 1,365 |
| `open_jev__entity-alignment-control-v1` | 1,282 |
| `followir_robust04` | 1,203 |
| `followir_news21` | 1,081 |

Query over 320, state fits (17):

| config | query tokens |
|---|---:|
| `trivia_qa` | 1,919 |
| `fineweb` | 1,354 |
| `agnews` | 1,265 |
| `pubmed` | 1,192 |
| `codesearch` | 1,056 |
| `bge-hn-hotpotqa` | 902 |
| `arxiv` | 834 |
| `wikipedia_hard_negatives_english` | 833 |
| `bge-hn-miracl_en` | 778 |
| `bge-hn-colliee` | 733 |
| `bge-hn-squad` | 718 |
| `natural_questions` | 687 |
| `oasst1` | 553 |
| `bge-hn-mr-tydi_english` | 519 |
| `bge-hn-pubmed_qa_labeled` | 404 |
| `bge-hn-nq` | 377 |
| `bge-hn-trivia` | 363 |

The training roster therefore drops from 153 to 124 configs. Three of the 32
(`followir_news21`, `followir_robust04`, `synthetic_relevance__nanocoir`) are
test-only and were never in the training roster; the other 29 are the
difference. Evaluation is untouched: `test_configs()` still returns all 106
benchmark configs, with the same truncation behaviour documented above.

To load them anyway, pass `include_unfit=True` to `train_configs()` or
`load_train()`, or set `SYSTEMONE_INCLUDE_UNFIT=1` for
`scripts/train_main.py`. Reproduce the list with
`uv run python scripts/measure_fit.py`; it also reports drift from the constant.

## One repo, three splits

Bekko is the only dataset dependency. Its 182 configs split three ways:

```
bekko train       153 configs   thousands of cases per config
bekko validation   82 configs   100 cases per config
bekko test        106 configs   byte-identical to the published benchmark
```

77 config names appear in both train and test: same task family, different rows.
The other 76 train configs have no benchmark counterpart at all. Bekko's
`validation` and `test` splits are the benchmark itself, so `split="validation"`
or `split="test"` on a training run is training on the test set wearing a
different name.

`load_train` asks for `split="train"`, so nothing in this project has leaked
that way. `train_configs()` and `test_configs()` are the two rosters; both read
the dataset card, so neither probes splits one config at a time.

## Where the files live

`datasets` caches outside the project:

```
~/.cache/huggingface/hub/datasets--hotchpotch--<name>/snapshots/<sha>/*.parquet
~/.cache/huggingface/datasets/hotchpotch___<name>/<config>/<version>/<sha>/*.arrow
```

The Arrow files are what `load_dataset` actually reads. First bekko load took
2,456 s (cold parquet download); the second took 609 s with no download.

## The old path: `hf.py`

`load_jevbench()` and `split_by_case()` are the earlier JevBench loader. It reads
a `questions` config and groups examples by `case_id` so that all decisions about
one case stay on the same side of a train/test split. Still used by
`scripts/train_legacy.py`; superseded by the bekko loaders.

Next: [Metrics →](10-metrics.md)
