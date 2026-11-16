import os
import time

import torch

from systemone import ModelConfig, build_model, checkpoint_path, load_or_build, save_checkpoint, set_seed
from systemone.data import filter_to_budget
from systemone.bpe import SentencePieceTokenizer, train_tokenizer
from systemone.calibrate import (
    accuracy,
    accuracy_by_kind,
    collect_logits,
    expected_calibration_error,
    fit_temperatures,
)
from systemone.s1mb import (
    BEKKO_REPO,
    all_configs,
    collect_texts,
    load_bekko_train,
    load_s1mb,
)
from systemone.train import evaluate, train

SEED = 0
NAME = os.environ.get("SYSTEMONE_NAME", "bekko_tiny")
MAX_CASES = int(os.environ.get("BEKKO_CASES", "1000"))  # per config, not total
CONFIG_LIMIT = int(os.environ.get("BEKKO_CONFIG_LIMIT", "0"))  # 0 = every config
EPOCHS = int(os.environ.get("SYSTEMONE_EPOCHS", "2"))
MAX_ROWS = int(os.environ.get("SYSTEMONE_MAX_ROWS", "256"))
STATE_TOKENS = int(os.environ.get("SYSTEMONE_STATE_TOKENS", "192"))
QUERY_TOKENS = int(os.environ.get("SYSTEMONE_QUERY_TOKENS", "64"))
CHECKPOINT = os.environ.get("SYSTEMONE_CHECKPOINT", "0") == "1"
BF16 = os.environ.get("SYSTEMONE_BF16", "0") == "1"
GRAD_ACCUM = int(os.environ.get("SYSTEMONE_GRAD_ACCUM", "1"))
RESUME = os.environ.get("SYSTEMONE_RESUME", "0") == "1"
FIT_ONLY = os.environ.get("SYSTEMONE_FIT_ONLY", "0") == "1"
DEV = os.environ.get("SYSTEMONE_DEV", "0") == "1"

if DEV:
    # Fast loop: fewer configs, fewer candidates per case, a probe eval each
    # epoch, and the full split only at the end. Its own name so it cannot
    # reuse the full run's tokenizer or data cache.
    NAME = f"{NAME}_dev"
    CONFIG_LIMIT = CONFIG_LIMIT or 5
    MAX_CASES = min(MAX_CASES, 200)
    MAX_ROWS = min(MAX_ROWS, 128)
    BF16 = True
    PROBE = 2000
    MAX_CANDIDATES = 4
else:
    PROBE = 0
    MAX_CANDIDATES = 8

set_seed(SEED)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

started = time.perf_counter()
configs = all_configs(BEKKO_REPO)
if CONFIG_LIMIT:
    configs = configs[:CONFIG_LIMIT]
train_grouped = load_bekko_train(
    configs=configs, max_candidates=MAX_CANDIDATES, max_cases=MAX_CASES, seed=SEED
)
train_set = [example for case in train_grouped.values() for example in case]
print(f"train load_s={time.perf_counter() - started:.1f} configs={len(configs)} "
      f"cases={len(train_grouped)} examples={len(train_set)}")

test_grouped = load_s1mb(all_configs(), seed=SEED)
test_set = [example for case in test_grouped.values() for example in case]

counts = {"noul": 0, "choice": 0, "score": 0}
for example in train_set:
    counts[type(example.question).__name__.lower()] += 1
print(f"train kinds={counts}")
print(f"test examples={len(test_set)}")

directory = checkpoint_path(NAME)
tokenizer_file = directory / "tokenizer.model"
if tokenizer_file.exists():
    # Reuse: retraining per run is the same work every time and changes the
    # vocabulary the dataset cache is keyed on.
    tokenizer = SentencePieceTokenizer(tokenizer_file)
    print(f"tokenizer reused vocab={tokenizer.vocab_size}")
else:
    tokenizer = train_tokenizer(collect_texts(train_grouped), directory / "tokenizer", vocab_size=2048)
    print(f"tokenizer vocab={tokenizer.vocab_size}")

cfg = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    ffn_hidden=352,
    max_state_tokens=STATE_TOKENS,
    max_query_tokens=QUERY_TOKENS,
)
model = build_model(cfg, SEED)
print(f"params={cfg.parameter_count():,} ({cfg.parameter_count() / 1e6:.2f}M)")
print(f"budget={STATE_TOKENS}/{QUERY_TOKENS} rows={MAX_ROWS} accum={GRAD_ACCUM} "
      f"checkpoint={CHECKPOINT} bf16={BF16}")

full_test_set = test_set
if FIT_ONLY:
    kept_train = filter_to_budget(train_set, tokenizer, cfg)
    kept_test = filter_to_budget(test_set, tokenizer, cfg)
    print(f"fit-only {len(kept_train)}/{len(train_set)} train examples "
          f"({100*len(kept_train)/len(train_set):.1f}%), "
          f"{len(kept_test)}/{len(test_set)} test ({100*len(kept_test)/len(test_set):.1f}%)")
    train_set, test_set = kept_train, kept_test

suffix = "_fit" if FIT_ONLY else ""
train_data = load_or_build(directory / f"bekko_train_cache{suffix}.pt", train_set, tokenizer, cfg)
test_data = load_or_build(directory / f"s1mb_test_cache{suffix}.pt", test_set, tokenizer, cfg)
# The same model measured both ways: the fit-only subset is the number worth
# reporting, the full split shows what truncation costs by comparison.
full_test_data = (
    load_or_build(directory / "s1mb_test_cache.pt", full_test_set, tokenizer, cfg)
    if FIT_ONLY
    else None
)
# Dev mode evaluates a fixed probe each epoch and the full split only at the end.
probe_data = (
    load_or_build(directory / "probe_cache.pt", test_set[:PROBE], tokenizer, cfg)
    if PROBE
    else test_data
)
print(
    f"cache train={len(train_data)} test={len(test_data)} "
    f"state_tokens={train_data.state_data.numel():,} "
    f"query_tokens={train_data.query_data.numel():,}"
)

state_file = directory / "training_state.pt"
resume = torch.load(state_file, weights_only=False) if (RESUME and state_file.exists()) else None
if resume:
    print(f"resuming from epoch {resume['epoch']} (target {EPOCHS})")

def save_state(epoch, state):
    torch.save(state, state_file)
    print(f"  saved state at epoch {epoch}")

before = evaluate(model, probe_data, device=device)
started = time.perf_counter()
history = train(
    model,
    train_data,
    epochs=EPOCHS,
    max_rows=MAX_ROWS,
    lr=2e-3,
    device=device,
    seed=SEED,
    warmup_fraction=0.05,
    eval_dataset=probe_data,
    eval_every=1,
    grad_accum=GRAD_ACCUM,
    checkpoint=CHECKPOINT,
    bf16=BF16,
    resume=resume,
    on_epoch_end=save_state,
)
print(f"train_s={time.perf_counter() - started:.1f}")
print(f"steps={len(history.losses)}")
print(f"eval_curve={[(e, round(v, 4)) for e, v in history.evals]}")
print(f"loss first={history.losses[0]:.4f} last={history.losses[-1]:.4f}")

after = evaluate(model, test_data, device=device)
scope = "fit-only" if FIT_ONLY else "full split"
print(f"eval before={before:.4f} after={after:.4f} ({scope})")
if full_test_data is not None:
    print(f"same model on the full split (truncated)={evaluate(model, full_test_data, device=device):.4f}")

train_collected = collect_logits(model, train_data, device=device)
collected = collect_logits(model, test_data, device=device)
print("train_acc", {k: round(v, 3) for k, v in accuracy_by_kind(train_collected).items()})
print("test_acc", {k: round(v, 3) for k, v in accuracy_by_kind(collected).items()})
print(f"accuracy={accuracy(collected):.3f}")
# fit on train: fitting on the eval split would report calibration the model
# was optimised for, which is leakage
temperatures = fit_temperatures(train_collected)
print(f"ece_before={expected_calibration_error(collected):.4f}")
print("temperatures", {k: round(v, 3) for k, v in temperatures.items()})
print(f"ece_after={expected_calibration_error(collected, temperatures):.4f}")

saved = save_checkpoint(
    NAME, model, tokenizer, temperatures=temperatures, seed=SEED,
    extra={"eval_loss": after, "train_cases": MAX_CASES},
)
print(f"saved {saved}")
