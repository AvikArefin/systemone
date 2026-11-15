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
from systemone.hf import load_jevbench
from systemone.schema import Choice, Score
from systemone.train import evaluate, train

SEED = 0
NAME = "jevbench_tiny"

set_seed(SEED)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

grouped = load_jevbench()
train_set, test_set = split_by_case(grouped, test_fraction=0.2, seed=SEED)

texts: list[str] = []
for case in grouped.values():
    for example in case:
        texts.append(example.state)
        texts.append(example.question.instructions)
        if isinstance(example.question, (Choice, Score)):
            texts.extend(example.question.criteria)

directory = checkpoint_path(NAME)
tokenizer = train_tokenizer(texts, directory / "tokenizer", vocab_size=2048)
print(f"tokenizer vocab={tokenizer.vocab_size} sep={tokenizer.sep_id}")

cfg = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=128,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    ffn_hidden=352,
    max_state_tokens=384,
    max_query_tokens=64,
)
model = build_model(cfg, SEED)
print(f"params={cfg.parameter_count():,} ({cfg.parameter_count() / 1e6:.2f}M)")

train_data = load_or_build(directory / "train_cache.pt", train_set, tokenizer, cfg)
test_data = load_or_build(directory / "test_cache.pt", test_set, tokenizer, cfg)

before = evaluate(model, test_data, device=device)
started = time.perf_counter()
history = train(
    model, train_data, epochs=8, max_rows=256, lr=2e-3, device=device, seed=SEED
)
print(f"train_s={time.perf_counter() - started:.1f}")
after = evaluate(model, test_data, device=device)
print(f"loss first={history.losses[0]:.4f} last={history.losses[-1]:.4f} min={min(history.losses):.4f}")
print(f"eval before={before:.4f} after={after:.4f}")

collected = collect_logits(model, test_data, device=device)
print("test_acc", {k: round(v, 3) for k, v in accuracy_by_kind(collected).items()})
print(f"accuracy={accuracy(collected):.3f}")
temperatures = fit_temperatures(collected)
print("temperatures", {k: round(v, 3) for k, v in temperatures.items()})
print(f"ece_after={expected_calibration_error(collected, temperatures):.4f}")

saved = save_checkpoint(
    NAME, model, tokenizer, temperatures=temperatures, seed=SEED, extra={"eval_loss": after}
)
print(f"saved {saved}")
