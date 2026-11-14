import random


from systemone import (
    ByteTokenizer,
    Choice,
    ModelConfig,
    Noul,
    Score,
    build_model,
    set_seed,
)
from systemone.calibrate import (
    accuracy,
    collect_logits,
    expected_calibration_error,
    fit_temperatures,
)
from systemone.data import EncodedSet, Example
from systemone.train import evaluate, train

TEAMS = ["billing", "security", "support"]
LEVELS = ["can wait a week", "this week", "today", "now"]
OPENERS = ["Customer note:", "Inbound ticket:", "Internal report:", "Account update:"]
CUES = {
    "billing": ["invoice", "refund", "charge", "payment"],
    "security": ["breach", "password", "phishing", "login attempt"],
    "support": ["how do i", "broken", "not working", "setup"],
}
URGENCY = {"now": 3, "today": 2, "week": 1}


def make_examples(count: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        team = rng.choice(TEAMS)
        cue = rng.choice(CUES[team])
        urgency_word = rng.choice(list(URGENCY))
        dangerous = rng.random() < 0.5
        action = "delete" if dangerous else "inspect"
        state = (
            f"{rng.choice(OPENERS)} {cue}. The customer needs this resolved "
            f"{urgency_word} and asks us to {action} the related records. "
            f"Requested team: {team}."
        )
        examples.append(
            Example(
                state=state,
                question=Noul(instructions="Is this action risky to run without a human approving it?"),
                label=dangerous,
            )
        )
        examples.append(
            Example(
                state=state,
                question=Choice(instructions="Which team should handle this?", criteria=TEAMS),
                label=team,
            )
        )
        examples.append(
            Example(
                state=state,
                question=Score(instructions="How urgent is this?", criteria=LEVELS),
                label=URGENCY[urgency_word],
            )
        )
    return examples


cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=176)
tokenizer = ByteTokenizer()
set_seed(0)
model = build_model(cfg, 0)

data = make_examples(240, seed=1)
split = int(len(data) * 0.8)
train_set, test_set = data[:split], data[split:]
train_data = EncodedSet.build(train_set, tokenizer, cfg)
test_data = EncodedSet.build(test_set, tokenizer, cfg)

before = evaluate(model, test_data)
history = train(model, train_data, epochs=40, batch_size=16, lr=3e-3)
after = evaluate(model, test_data)
print(f"params={cfg.parameter_count():,}")
print(f"loss first={history.losses[0]:.4f} last={history.losses[-1]:.4f}")
print(f"eval before={before:.4f} after={after:.4f}")

collected = collect_logits(model, test_data)
print(f"accuracy={accuracy(collected):.3f}")
print(f"ece_before={expected_calibration_error(collected):.4f}")

temperatures = fit_temperatures(collected)
print("temperatures", {k: round(v, 3) for k, v in temperatures.items()})
print(f"ece_after={expected_calibration_error(collected, temperatures):.4f}")
