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
| [`calibrate.py`](../src/systemone/calibrate.py) | Metrics, temperature fitting, ECE | `collect_logits`, `fit_temperatures`, `expected_calibration_error` |
| [`inference.py`](../src/systemone/inference.py) | One request in, typed answers out | `answer_request` |
| [`checkpoint.py`](../src/systemone/checkpoint.py) | Save/load a named model directory | `save_checkpoint`, `load_checkpoint` |
| [`paths.py`](../src/systemone/paths.py) | Where checkpoints live | `CHECKPOINT_DIR`, `checkpoint_path()` |
| [`seeding.py`](../src/systemone/seeding.py) | Reproducibility | `set_seed`, `build_model` |
| [`s1mb.py`](../src/systemone/s1mb.py) | S1MB + bekko corpus loaders and row parsing | `load_typed_decisions`, `load_s1mb`, `load_bekko_train`, `parse_decision` |
| [`hf.py`](../src/systemone/hf.py) | Legacy JevBench loader | `load_jevbench`, `split_by_case` |
| [`__init__.py`](../src/systemone/__init__.py) | Public surface | re-exports |

## Scripts — `scripts/`

| Script | Purpose | Typical runtime |
|---|---|---|
| [`smoke.py`](../scripts/smoke.py) | Construct a model, run one request, check shapes | seconds |
| [`train_demo.py`](../scripts/train_demo.py) | Synthetic 3-task demo, end-to-end with calibration | ~30 s |
| [`train_bekko.py`](../scripts/train_bekko.py) | Train on the bekko corpus, evaluate on S1MB test | 20 min – 2 h |
| [`train_s1mb.py`](../scripts/train_s1mb.py) | Train on S1MB configs (contaminates the benchmark — comparison only) | ~40 min |
| [`train_jevbench_bpe.py`](../scripts/train_jevbench_bpe.py) | Train on JevBench with a trained BPE tokenizer | ~20 min |

## The data flow, end to end

```
HF dataset  (input.state_json, input.decisions, targets)
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
| Change the prompt template | `prompt.py` | `_render` is the only place |
| Change loss weighting | `data.py` | `batch_loss` — currently equal per group |
| Change LR/schedule | `train.py` | `warmup_fraction`, `min_lr_ratio` |
| Add a new decision type | `schema.py` → `data.rows_for` → `batch_loss` → `inference` | four places, in that order |
| Swap the tokenizer | `tokenizer.py` / `bpe.py` | keep `pad_id`/`sep_id` meaning |
| Add a dataset | `s1mb.py` | `load_typed_decisions` expects the shared schema |
| Verify a checkpoint still loads | `checkpoint.py` | keys are the same as training |

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
