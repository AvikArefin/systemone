import os
import time

import torch

from systemone import ModelConfig, build_model, checkpoint_path, load_or_build, save_checkpoint, set_seed
from systemone.bpe import train_tokenizer
from systemone.calibrate import (
    accuracy,
    accuracy_by_kind,
    collect_logits,
    expected_calibration_error,
    fit_temperatures,
)
from systemone.s1mb import all_configs, collect_texts, load_bekko_train, load_s1mb
from systemone.train import evaluate, train

SEED = 0
NAME = "bekko_tiny"
MAX_CASES = int(os.environ.get("BEKKO_CASES", "1000"))
EPOCHS = int(os.environ.get("SYSTEMONE_EPOCHS", "2"))
BATCH = int(os.environ.get("SYSTEMONE_BATCH", "32"))

set_seed(SEED)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

started = time.perf_counter()
train_grouped = load_bekko_train(max_cases=MAX_CASES, seed=SEED)
train_set = [example for case in train_grouped.values() for example in case]
print(f"train load_s={time.perf_counter() - started:.1f} cases={len(train_grouped)} examples={len(train_set)}")

test_grouped = load_s1mb(all_configs(), seed=SEED)
test_set = [example for case in test_grouped.values() for example in case]

counts = {"noul": 0, "choice": 0, "score": 0}
for example in train_set:
    counts[type(example.question).__name__.lower()] += 1
print(f"train kinds={counts}")
print(f"test examples={len(test_set)}")

directory = checkpoint_path(NAME)
tokenizer = train_tokenizer(collect_texts(train_grouped), directory / "tokenizer", vocab_size=2048)
print(f"tokenizer vocab={tokenizer.vocab_size}")

cfg = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    ffn_hidden=352,
    max_state_tokens=192,
    max_query_tokens=48,
)
model = build_model(cfg, SEED)
print(f"params={cfg.parameter_count():,} ({cfg.parameter_count() / 1e6:.2f}M)")

train_data = load_or_build(directory / "bekko_train_cache.pt", train_set, tokenizer, cfg)
test_data = load_or_build(directory / "s1mb_test_cache.pt", test_set, tokenizer, cfg)
print(
    f"cache train={len(train_data)} test={len(test_data)} "
    f"state_tokens={train_data.state_data.numel():,} "
    f"query_tokens={train_data.query_data.numel():,}"
)

before = evaluate(model, test_data, device=device)
started = time.perf_counter()
history = train(
    model,
    train_data,
    epochs=EPOCHS,
    batch_size=BATCH,
    lr=2e-3,
    device=device,
    seed=SEED,
    warmup_fraction=0.05,
    eval_dataset=test_data,
    eval_every=1,
)
print(f"train_s={time.perf_counter() - started:.1f}")
print(f"steps={len(history.losses)}")
print(f"eval_curve={[(e, round(v, 4)) for e, v in history.evals]}")
print(f"loss first={history.losses[0]:.4f} last={history.losses[-1]:.4f}")

after = evaluate(model, test_data, device=device)
print(f"eval before={before:.4f} after={after:.4f}")

train_collected = collect_logits(model, train_data, device=device)
collected = collect_logits(model, test_data, device=device)
print("train_acc", {k: round(v, 3) for k, v in accuracy_by_kind(train_collected).items()})
print("test_acc", {k: round(v, 3) for k, v in accuracy_by_kind(collected).items()})
print(f"accuracy={accuracy(collected):.3f}")
print(f"ece_before={expected_calibration_error(collected):.4f}")
temperatures = fit_temperatures(collected)
print("temperatures", {k: round(v, 3) for k, v in temperatures.items()})
print(f"ece_after={expected_calibration_error(collected, temperatures):.4f}")

saved = save_checkpoint(
    NAME, model, tokenizer, temperatures=temperatures, seed=SEED,
    extra={"eval_loss": after, "train_cases": MAX_CASES},
)
print(f"saved {saved}")
