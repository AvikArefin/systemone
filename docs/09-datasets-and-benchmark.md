# 09 — Datasets and benchmark

Where the data comes from, what shape it is in, and the contamination trap.
Files: [`s1mb.py`](../src/systemone/s1mb.py), [`hf.py`](../src/systemone/hf.py).

## The format everyone shares

S1MB and the bekko training corpus use **the same schema**, which is why one
parser handles both:

```
input.state_json      the shared state, as JSON
input.decisions[]     typed questions: id, type (noul/choice/score),
                      instructions_json, criteria[], documents[]
targets[]             aligned distributions: decision_id, ids[], probabilities[]
split, case_id, group_id, language, input_hash
```

`probabilities` are soft: `[0.7, 0.3]` is a real target, not an error. Keeping
candidate order and soft targets intact is explicit in both dataset cards.

## Evaluation: S1MB

**System One Mosaic Benchmark**, `hotchpotch/s1mb-dataset` (renamed from
`S1MB-english-v1`).

| | |
|---|---|
| Configs (subsets) | 106 |
| Cases | 14,009 |
| Judgments | 26,269 |
| Splits | `test` only |
| Benchmarks | 137 (59 noul, 57 choice, 21 score) |

Homepage numbers matter here: **the distributed data is an evaluation-only test
split**. There is no train split and no validation split. If you train on it, you
are training on the benchmark.

`load_s1mb()` reads it with `split="test"`.

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

`load_bekko_train(max_cases=N)` caps rows per subset, because a full pass is
~262k steps at batch 32 — about 27 hours on this machine.

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

Our first "improvement" was training on the S1MB test split. That produced
0.417 → 0.551 and felt like progress. It was not comparable to anything:

- Published S1MB scores come from models that never trained on it.
- Our number came from a model that trained on 80% of it.
- Even after switching to the bekko corpus we still share **dataset families**
  with the evaluation — bekko's own authors disclose 77 of 106 subset names in
  common. Familiarity with a task family inflates scores without proving
  generalisation.

What we do now:

1. Train on `bekko-system-one-dataset-v0`, never on S1MB.
2. Evaluate on S1MB test with rows held out (different rows, same families).
3. Say "same-family exposure" out loud whenever quoting a number.
4. Treat S1MB's six synthetic generalisation subsets as the honest signal, since
   they test instruction-following rather than family familiarity.

## The state budget vs. the real corpus

`max_state_tokens` is 192. Measured on the S1MB test split, one row per case,
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
with the `s1mb_tiny` tokenizer, which is a different BPE vocabulary over the same
text. Compare budgets within one tokenizer, never across.

Bumping this budget is a cheap accuracy win that no run so far has tested — every
headline number in [Experiments](11-experiments.md) was produced at 192. The cost
is quadratic attention, so it is a real trade-off rather than a free fix, but it
should be measured before "capacity" is blamed. A 768-token state still only
leaves 18.1% of state tokens behind, versus 72.7% at 192.

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
`scripts/train_jevbench_bpe.py`; superseded by the S1MB corpus loaders.

Next: [Metrics →](10-metrics.md)
