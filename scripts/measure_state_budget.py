"""Measure what the state-token cap costs on the test split.

Loads a checkpoint, re-encodes the test split at several `max_state_tokens`
values, and scores each with the baseline-adjusted metric. RoPE makes the
model length-agnostic, so no retraining is needed -- but the weights never
saw more than the trained cap, so a flat result means training, not the cap,
is the limit.

Reports the whole split and the subset that the trained cap actually cut.
Run it after any change to `max_state_tokens` so the numbers in the docs stay
reproducible.
"""

import json
import os
import sys
import time
from dataclasses import replace

import torch

from systemone import checkpoint_path, load_checkpoint, set_seed
from systemone.data import EncodedSet
from systemone.corpus import load_test, test_configs
from systemone.scoring import Decision, build_report, probabilities_by_group

NAME = os.environ.get("SYSTEMONE_STATE_CHECKPOINT", "bekko_tiny")
CONFIGS = int(os.environ.get("SYSTEMONE_STATE_CONFIGS", "0")) or None
BUDGETS = [
    int(value)
    for value in os.environ.get("SYSTEMONE_STATE_BUDGETS", "192,384,512").split(",")
]
QUERY_BUDGETS = [
    int(value)
    for value in os.environ.get("SYSTEMONE_QUERY_BUDGETS", "64").split(",")
]
MAX_ROWS = int(os.environ.get("SYSTEMONE_STATE_ROWS", "256"))
BASE = 192

set_seed(0)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model, tokenizer, temperatures = load_checkpoint(NAME)
print(f"checkpoint={NAME} params={model.cfg.parameter_count():,} device={device}")


configs = sys.argv[1:] or test_configs()
if CONFIGS is not None:
    configs = configs[:CONFIGS]

grouped = load_test(
    configs, max_candidates=None, seed=0, shuffle_options=False, strict=True
)
examples = [example for case in grouped.values() for example in case]
if not examples:
    sys.exit(f"no examples loaded from {len(configs)} configs")
state_lengths = [len(tokenizer.encode(example.state)) for example in examples]
truncated = [index for index, length in enumerate(state_lengths) if length > BASE]
print(
    f"configs={len(configs)} cases={len(grouped)} examples={len(examples)} "
    f"truncated_at_{BASE}={len(truncated)}"
)


def decisions_for(probabilities: list[list[float]]) -> list[Decision]:
    decisions = []
    for example, probs in zip(examples, probabilities):
        kind = example.question.__class__.__name__.lower()
        if kind == "choice":
            label = float(example.question.criteria.index(example.label))
        else:
            label = float(example.label)
        decisions.append(
            Decision(
                benchmark=example.benchmark or "unknown",
                kind=kind,
                probabilities=probs,
                distribution=example.distribution,
                label=label,
                candidate_ids=example.candidate_ids,
                values=example.values,
            )
        )
    return decisions


def summary(report) -> str:
    scores = report.task_scores
    return (
        f"{100 * scores['noul']:>6.2f}{100 * scores['choice']:>8.2f}"
        f"{100 * scores['score']:>7.2f}{100 * report.task_average:>8.2f}"
    )


print()
print(f"{'state/qry':>10}{'cut':>7}  {'noul':>6}{'choice':>8}{'score':>7}{'avg':>8}"
      f"  {'noul':>6}{'choice':>8}{'score':>7}{'avg':>8}")
print(f"{'':>17}  {'-------- whole split --------':>29}  {'----- states over 192 -----':>31}")

records = []
for state_budget in BUDGETS:
    cut_states = sum(1 for length in state_lengths if length > state_budget)
    for query_budget in QUERY_BUDGETS:
        cfg = replace(
            model.cfg,
            vocab_size=tokenizer.vocab_size,
            max_state_tokens=state_budget,
            max_query_tokens=query_budget,
        )
        started = time.perf_counter()
        dataset = EncodedSet.build(examples, tokenizer, cfg)
        probabilities = probabilities_by_group(
            model, dataset, device=device, max_rows=MAX_ROWS, temperatures=temperatures
        )
        elapsed = time.perf_counter() - started

        decisions = decisions_for(probabilities)
        whole = build_report(decisions)
        cut = build_report([decisions[index] for index in truncated])

        print(
            f"{f'{state_budget}/{query_budget}':>10}{cut_states:>7}  "
            f"{summary(whole)}  {summary(cut)}"
        )
        records.append(
            {
                "state_budget": state_budget,
                "query_budget": query_budget,
                "states_over_budget": cut_states,
                "seconds": round(elapsed, 1),
                "whole": whole.task_scores | {"task_average": whole.task_average * 100},
                "cut": cut.task_scores | {"task_average": cut.task_average * 100},
                "cut_excluded": cut.excluded,
                "whole_excluded": whole.excluded,
            }
        )

out = checkpoint_path(NAME) / "state_budget_report.json"
out.write_text(json.dumps({"checkpoint": NAME, "results": records}, indent=2))
print()
print(f"wrote {out}")
