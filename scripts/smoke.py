import torch

from systemone import ByteTokenizer, Choice, ModelConfig, Noul, Request, Score, TinyJev, answer_request, set_seed

cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=176)
tokenizer = ByteTokenizer()
set_seed(0)
model = TinyJev(cfg).eval()

print(f"params={cfg.parameter_count():,} kv_bytes@192={cfg.kv_cache_bytes(192):,}")

request = Request(
    state=(
        "Task: clean up inactive accounts before the quarterly report. "
        "The script deletes accounts with no logins in 90 days across billing and support tenants."
    ),
    questions={
        "safe": Noul(instructions="Is this action safe to run without a human approving it first?"),
        "queue": Choice(
            instructions="Which team should handle this?",
            criteria=["billing", "security", "support"],
        ),
        "urgency": Score(
            instructions="How urgent is this?",
            criteria=["Can wait a week", "This week", "Today", "Now"],
        ),
    },
)

answers = answer_request(model, tokenizer, request)
for name, answer in answers.items():
    print(name, answer)

with torch.no_grad():
    ids = torch.tensor([tokenizer.encode("hello world")], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    logits = model(ids, mask, ids, mask)
print("shape_ok", tuple(logits.shape))
