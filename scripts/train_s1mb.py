import os
import time

import torch

from systemone import (
    ModelConfig,
    build_model,
    checkpoint_path,
    load_or_build,
    save_checkpoint,
    set_seed,
    split_by_case,
)
from systemone.bpe import train_tokenizer
from systemone.calibrate import (
    accuracy,
    accuracy_by_kind,
    collect_logits,
    expected_calibration_error,
    fit_temperatures,
)
from systemone.s1mb import all_configs, collect_texts, load_s1mb
from systemone.train import evaluate, train

SEED = 0
NAME = "s1mb_tiny"
EPOCHS = int(os.environ.get("SYSTEMONE_EPOCHS", "10"))

CONFIGS = all_configs()

set_seed(SEED)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
grouped = load_s1mb(CONFIGS, seed=SEED)
counts = {"noul": 0, "choice": 0, "score": 0}
for case in grouped.values():
    for example in case:
        counts[type(example.question).__name__.lower()] += 1
print(f"cases={len(grouped)} examples={sum(len(v) for v in grouped.values())} {counts}")
print("soft_examples", sum(1 for v in grouped.values() for e in v if e.distribution is not None))

train_set, test_set = split_by_case(grouped, test_fraction=0.2, seed=SEED)
print(f"train_rows={len(train_set)} test_rows={len(test_set)}")

directory = checkpoint_path(NAME)
tokenizer = train_tokenizer(collect_texts(grouped), directory / "tokenizer", vocab_size=2048)
print(f"tokenizer vocab={tokenizer.vocab_size}")

cfg = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    ffn_hidden=352,
    max_state_tokens=192,
    max_query_tokens=64,
)
model = build_model(cfg, SEED)
print(f"params={cfg.parameter_count():,} ({cfg.parameter_count() / 1e6:.2f}M)")

train_data = load_or_build(directory / "train_cache.pt", train_set, tokenizer, cfg)
test_data = load_or_build(directory / "test_cache.pt", test_set, tokenizer, cfg)
print(
    f"cache examples={len(train_data)}/{len(test_data)} "
    f"state_tokens={train_data.state_data.numel():,} "
    f"query_tokens={train_data.query_data.numel():,}"
)

before = evaluate(model, test_data, device=device)
started = time.perf_counter()
history = train(
    model,
    train_data,
    epochs=EPOCHS,
    batch_size=32,
    lr=2e-3,
    device=device,
    seed=SEED,
    warmup_fraction=0.05,
    eval_dataset=test_data,
    eval_every=2,
)
print(f"train_s={time.perf_counter() - started:.1f}")
after = evaluate(model, test_data, device=device)
if history.losses:
    print(f"steps={len(history.losses)} lr_first={history.lrs[0]:.2e} lr_last={history.lrs[-1]:.2e}")
    print(f"eval_curve={[(e, round(v, 4)) for e, v in history.evals]}")
    print(f"loss first={history.losses[0]:.4f} last={history.losses[-1]:.4f} min={min(history.losses):.4f}")
print(f"eval before={before:.4f} after={after:.4f}")

collected = collect_logits(model, test_data, device=device)
train_collected = collect_logits(model, train_data, device=device)
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
    NAME,
    model,
    tokenizer,
    temperatures=temperatures,
    seed=SEED,
    extra={"configs": CONFIGS, "eval_loss": after},
)
print(f"saved {saved}")
