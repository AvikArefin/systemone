"""Score a checkpoint on S1MB with the baseline-adjusted Task Avg metric.

Unlike `scripts/train_*.py`, this keeps the benchmark's own candidate sets and
order (`max_candidates=None`, `shuffle_options=False`) and refuses to drop a
decision silently, so the number is comparable to the published table.
"""

import json
import os

import torch

from systemone import ModelConfig, checkpoint_path, load_checkpoint, set_seed
from systemone.bpe import SentencePieceTokenizer
from systemone.data import EncodedSet
from systemone.s1mb import all_configs, load_s1mb
from systemone.scoring import Decision, build_report, probabilities_by_group

NAME = os.environ.get("SYSTEMONE_EVAL_CHECKPOINT", "bekko_tiny")
SEED = 0
LIMIT = int(os.environ.get("SYSTEMONE_EVAL_CONFIGS", "0")) or None

set_seed(SEED)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

directory = checkpoint_path(NAME)
model, tokenizer, temperatures = load_checkpoint(NAME)
print(f"checkpoint={NAME} params={model.cfg.parameter_count():,} device={device}")
print(f"temperatures={ {k: round(v, 3) for k, v in temperatures.items()} }")

configs = all_configs()
if LIMIT is not None:
    configs = configs[:LIMIT]

grouped = load_s1mb(
    configs, max_candidates=None, seed=SEED, shuffle_options=False, strict=True
)
examples = [example for case in grouped.values() for example in case]
print(f"configs={len(configs)} cases={len(grouped)} examples={len(examples)}")

cfg = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=model.cfg.d_model,
    n_layers=model.cfg.n_layers,
    n_heads=model.cfg.n_heads,
    n_kv_heads=model.cfg.n_kv_heads,
    ffn_hidden=model.cfg.ffn_hidden,
    max_state_tokens=model.cfg.max_state_tokens,
    max_query_tokens=model.cfg.max_query_tokens,
)
cache = directory / "s1mb_eval_cache.pt"
dataset = EncodedSet.build(examples, tokenizer, cfg)
dataset.save(cache)
print(f"rows={len(dataset.query_offsets) - 1} cache={cache.name}")

probabilities = probabilities_by_group(
    model, dataset, device=device, temperatures=temperatures
)

decisions = []
for index, example in enumerate(examples):
    kind = example.question.__class__.__name__.lower()
    question = example.question
    if kind == "choice":
        label = float(question.criteria.index(example.label))
    else:
        label = float(example.label)
    decisions.append(
        Decision(
            benchmark=example.benchmark or "unknown",
            kind=kind,
            probabilities=probabilities[index],
            distribution=example.distribution,
            label=label,
            candidate_ids=example.candidate_ids,
            values=example.values,
        )
    )

report = build_report(decisions)

print()
print(f"{'benchmark':<52}{'kind':>8}{'n':>7}{'raw':>8}{'base':>8}{'adj':>8}")
for score in sorted(report.benchmarks, key=lambda s: (s.kind, s.task)):
    baseline = "-" if score.baseline is None else f"{score.baseline:.3f}"
    print(
        f"{score.task[:50]:<52}{score.kind:>8}{score.decisions:>7}"
        f"{score.raw:>8.3f}{baseline:>8}{score.adjusted:>8.3f}"
    )

print()
for kind, value in report.task_scores.items():
    print(f"{kind:>8} task score = {value * 100:.2f}")
print(f"{'Task Avg':>8}            = {report.task_average * 100:.2f}")
if report.excluded:
    print(f"excluded benchmarks (no headroom): {report.excluded}")

out = directory / "s1mb_report.json"
out.write_text(
    json.dumps(
        {
            "checkpoint": NAME,
            "parameters": model.cfg.parameter_count(),
            "task_scores": report.task_scores,
            "task_average": report.task_average * 100,
            "excluded": report.excluded,
            "benchmarks": [score.__dict__ for score in report.benchmarks],
        },
        indent=2,
    )
)
print(f"wrote {out}")
