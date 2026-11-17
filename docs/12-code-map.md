# 12 — Code map

Every file, what it does, and where to look when you want to change something.

## Library — `src/systemone/`

| File | Job | Key names |
|---|---|---|
| [`config.py`](../src/systemone/config.py) | Architecture hyperparameters + exact parameter/KV maths | `ModelConfig`, `parameter_count()`, `kv_cache_bytes()` |
| [`model.py`](../src/systemone/model.py) | The network | `TinyJev`, `Block`, `Attention`, `_rope` |
| [`schema.py`](../src/systemone/schema.py) | Typed questions and answers, with validation | `Noul`, `Choice`, `Score`, `Request`, `make_choice()` |
| [`data.py`](../src/systemone/data.py) | Examples → rows → flat cache → padded batch; loss | `Example`, `rows_for`, `EncodedSet`, `Batch`, `batch_loss`, `load_or_build` |
| [`tokenizer.py`](../src/systemone/tokenizer.py) | Tokenizer protocol, byte tokenizer, truncation | `ByteTokenizer`, `truncate_head_tail`, `PAD_ID`, `SEP_ID` |
| [`bpe.py`](../src/systemone/bpe.py) | SentencePiece training and wrapper | `train_tokenizer`, `SentencePieceTokenizer` |
| [`prompt.py`](../src/systemone/prompt.py) | Turns a question + candidate into token IDs | `choice_query`, `noul_query`, `score_query` |
| [`train.py`](../src/systemone/train.py) | Training loop, schedule, evaluation | `train`, `evaluate`, `History` |
| [`calibrate.py`](../src/systemone/calibrate.py) | Metrics, temperature fitting, ECE | `collect_logits`, `fit_temperatures`, `is_identifiable`, `expected_calibration_error` |
| [`scoring.py`](../src/systemone/scoring.py) | Baseline-adjusted test scoring (Task Avg) | `build_report`, `probabilities_by_group`, `Decision` |
| [`inference.py`](../src/systemone/inference.py) | One request in, typed answers out | `answer_request` |
| [`checkpoint.py`](../src/systemone/checkpoint.py) | Save/load a named model directory | `save_checkpoint`, `load_checkpoint` |
| [`paths.py`](../src/systemone/paths.py) | Where checkpoints live | `CHECKPOINT_DIR`, `checkpoint_path()` |
| [`seeding.py`](../src/systemone/seeding.py) | Reproducibility | `set_seed`, `build_model` |
| [`corpus.py`](../src/systemone/corpus.py) | Dataset rosters and row loading for train, validation and test | `load_typed_decisions`, `load_train`, `load_test`, `train_configs`, `test_configs`, `UNFIT_CONFIGS`, `parse_decision` |
| [`decontamination.py`](../src/systemone/decontamination.py) | Test-duplicate fingerprints and the blocklist | `case_fingerprint`, `find_test_duplicates`, `load_blocklist`, `write_blocklist` |
| [`hf.py`](../src/systemone/hf.py) | Legacy JevBench loader | `load_jevbench`, `split_by_case` |
| [`__init__.py`](../src/systemone/__init__.py) | Public surface | re-exports |

## Scripts — `scripts/`

| Script | Purpose | Typical runtime |
|---|---|---|
| [`smoke.py`](../scripts/smoke.py) | Construct a model, run one request, check shapes | seconds |
| [`train_synthetic.py`](../scripts/train_synthetic.py) | Synthetic 3-task demo, end-to-end with calibration | ~30 s |
| [`train_main.py`](../scripts/train_main.py) | Train on the train split, evaluate on test | 20 min – 2 h |
| [`train_comparison.py`](../scripts/train_comparison.py) | Train on the benchmark's test split (contaminates it — comparison only) | ~40 min |
| [`train_legacy.py`](../scripts/train_legacy.py) | Train on JevBench with a trained BPE tokenizer | ~20 min |
| [`measure_budget.py`](../scripts/measure_budget.py) | Measure what each query budget preserves on the test split, per decision kind | minutes |
| [`measure_state_budget.py`](../scripts/measure_state_budget.py) | Score a checkpoint at several state/query budgets, whole split and on the rows the cap cut | 5 – 30 min |
| [`measure_fit.py`](../scripts/measure_fit.py) | Report configs that overflow 1024/320 and drift from `UNFIT_CONFIGS` | seconds |
| [`measure_attention_memory.py`](../scripts/measure_attention_memory.py) | Mask tensor size and attention time per mask strategy | seconds |
| [`eval.py`](../scripts/eval.py) | Baseline-adjusted Task Avg on the test split | minutes |
| [`build_tokenizers.py`](../scripts/build_tokenizers.py) | Rebuild and measure comparison tokenizers | minutes |
| [`build_blocklist.py`](../scripts/build_blocklist.py) | Scan train for test duplicates and write the blocklist | one full pass over train |
| [`browse.py`](../scripts/browse.py) | Local browser for cached rows, prompts and tokens | instant |

## The data flow, end to end

```
bekko (HF dataset: input.state_json, input.decisions, targets)
    │  load_typed_decisions → parse_decision
    ▼
Example(state, question, label, distribution, benchmark, ids, values)
    │  rows_for(example, tokenizer)
    ▼
rows: list[list[int]]  +  target  +  kind
    │  EncodedSet.build  (once, saved to disk)
    ▼
flat int32 buffers + offset arrays
    │  EncodedSet.collate(batch_indices)
    ▼
Batch(state_ids, state_mask, query_ids, query_mask, group_ids, targets, kinds, distributions)
    │  TinyJev.forward
    ▼
logits [R]  ──►  batch_loss (per group)  ──►  backward
                     │
                     └──►  collect_logits ──► fit_temperatures ──► checkpoint
```

## "I want to change X" — where to go

| Goal | File | Note |
|---|---|---|
| Make the model bigger/smaller | `config.py` | then `parameter_count()` tells you the cost |
| Change head count or GQA ratio | `config.py` | `n_heads % n_kv_heads == 0` and even `head_dim` |
| Change the token budget | `config.py` | `max_state_tokens`, `max_query_tokens` |
| Change how many rows go in a batch | `data.py` | `EncodedSet.batches(max_rows)` — rows, never examples; a choice carries one row per candidate |
| Change where the state tower runs | `model.py` | `TinyJev.forward(..., group_ids)` — one state per group, not per row |
| Change the prompt template | `prompt.py` | `_render` is the only place |
| Change loss weighting | `data.py` | `batch_loss` — currently equal per group |
| Change LR/schedule | `train.py` | `warmup_fraction`, `min_lr_ratio` |
| Add a new decision type | `schema.py` → `data.rows_for` → `batch_loss` → `inference` | four places, in that order |
| Swap the tokenizer | `tokenizer.py` / `bpe.py` | keep `pad_id`/`sep_id` meaning |
| Add a dataset | `corpus.py` | `load_typed_decisions` expects the shared schema |
| Verify a checkpoint still loads | `checkpoint.py` | keys are the same as training |

## Tests — `tests/`

| File | Guards |
|---|---|
| [`test_encoding.py`](../tests/test_encoding.py) | A row can answer its own question: candidate never deleted, options stay distinct |
| [`test_data_pipeline.py`](../tests/test_data_pipeline.py) | Flat-cache offsets, group membership, per-group loss, cache versioning |
| [`test_model_invariants.py`](../tests/test_model_invariants.py) | Prefix mask, causal queries, padding invariance, pooling, RoPE, parameter count |
| [`test_scoring.py`](../tests/test_scoring.py) | The three adjusted metrics return 0 for trivial policies and 1 for perfect |
| [`test_schema_and_parsing.py`](../tests/test_schema_and_parsing.py) | Validation, wide option sets, dataset parsing |
| [`test_training_and_calibration.py`](../tests/test_training_and_calibration.py) | Loss decreases, schedule shape, ECE, no test-set leakage |
| [`test_inference.py`](../tests/test_inference.py) | Serving encodes rows exactly as training does |
| [`test_checkpoint.py`](../tests/test_checkpoint.py) | Save/load round trip, temperatures, metadata |
| [`test_regressions.py`](../tests/test_regressions.py) | One test per bug that shipped, with the damage recorded |
| [`test_dataset_integration.py`](../tests/test_dataset_integration.py) | Real corpus shapes (marked `slow`, skipped without the HF cache) |

```bash
uv run pytest           # fast suite (~3 s)
uv run pytest -m slow   # dataset-backed (~9 min, needs the HF cache)
```

The regression file is the institutional memory: each class names a bug, quantifies
what it cost, and asserts the property that was violated.

## Invariants to preserve

Break these and silent wrongness follows:

1. **`pad_id` is 0 and never a real token.** Masks are `ids != pad_id`.
2. **Candidate order is the answer.** Softmax indices, target indices, and
   `candidate_ids` must stay aligned.
3. **The state segment never attends to the query segment.** That is the prefix
   property the whole design relies on.
4. **One logit per row, softmax outside the model.** Putting a softmax inside
   would break `noul` and the calibration path.
5. **Cache fingerprints must match the tokenizer.** Same `vocab_size` is not
   enough; IDs differ after retraining.
6. **Positions restart per segment** (state `0..S-1`, query `0..T-1`). If you
   change this, re-train — it is not a free refactor.

Next: [Deployment →](13-deployment.md)
