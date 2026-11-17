"""Measure what each query budget preserves on real test rows.

Reports, per decision kind and per budget, how much of the instruction and the
candidate survive `_split_budget`, and how many whole rows survive. Run it after
any change to `max_query_tokens` so the numbers in the docs stay reproducible.
"""

import os
import sys
from collections import defaultdict

from systemone import checkpoint_path
from systemone.bpe import SentencePieceTokenizer
from systemone.prompt import _split_budget
from systemone.corpus import BEKKO_REPO, load_typed_decisions, test_configs

NAME = os.environ.get("SYSTEMONE_BUDGET_CHECKPOINT", "bekko_tiny")
MAX_CASES = int(os.environ.get("SYSTEMONE_BUDGET_CASES", "0")) or None
BUDGETS = tuple(
    int(value)
    for value in os.environ.get("SYSTEMONE_BUDGETS", "48,64,96,128,192,256,384,512,768").split(",")
)


def percentiles(values: list[int]) -> str:
    ordered = sorted(values)
    pick = lambda p: ordered[min(len(ordered) - 1, int(len(ordered) * p))]
    return f"p50={pick(0.5)} p90={pick(0.9)} p99={pick(0.99)} max={ordered[-1]}"


tokenizer = SentencePieceTokenizer(checkpoint_path(NAME) / "tokenizer.model")
configs = sys.argv[1:] or test_configs()
print(f"checkpoint={NAME} configs={len(configs)} max_cases={MAX_CASES or 'all'}")

grouped = load_typed_decisions(
    BEKKO_REPO,
    configs,
    split="test",
    max_cases=MAX_CASES,
    max_candidates=None,
    shuffle_options=False,
)
examples = [example for case in grouped.values() for example in case]
print(f"cases={len(grouped)} decisions={len(examples)}")

rows: dict[tuple[int, str], list[tuple[float, float | None, bool, int]]] = defaultdict(list)
lengths: dict[str, list[int]] = defaultdict(list)

for example in examples:
    question = example.question
    kind = type(question).__name__.lower()
    instruction_len = len(tokenizer.encode(question.instructions))
    candidates = [None] if kind == "noul" else [
        len(tokenizer.encode(option)) for option in question.criteria
    ]
    for candidate_len in candidates:
        if candidate_len is not None:
            lengths[kind].append(instruction_len + 1 + candidate_len)
        for budget in BUDGETS:
            if candidate_len is None:
                instruction_budget, candidate_budget = budget, None
            else:
                instruction_budget, candidate_budget = _split_budget(
                    budget, instruction_len, candidate_len
                )
            rows[(budget, kind)].append(
                (
                    min(instruction_budget, instruction_len) / instruction_len,
                    None if candidate_budget is None else min(candidate_budget, candidate_len) / candidate_len,
                    instruction_budget >= instruction_len
                    and (candidate_budget is None or candidate_budget >= candidate_len),
                )
            )

print()
print(f"{'budget':>6}  {'kind':<7}{'rows':>7}{'cut':>7}{'instr kept':>12}{'cand kept':>11}{'whole rows':>12}")
for budget in BUDGETS:
    for kind in ("noul", "choice", "score"):
        results = rows.get((budget, kind))
        if not results:
            continue
        count = len(results)
        cut = sum(1 for r in results if not r[2])
        instruction = sum(r[0] for r in results) / count
        candidate_values = [r[1] for r in results if r[1] is not None]
        candidate = (
            f"{100 * sum(candidate_values) / len(candidate_values):.1f}%"
            if candidate_values
            else "-"
        )
        whole = sum(1 for r in results if r[2]) / count
        print(
            f"{budget:>6}  {kind:<7}{count:>7}{100 * cut / count:>6.1f}%"
            f"{100 * instruction:>11.1f}%{candidate:>11}{100 * whole:>11.1f}%"
        )
    print()

print("row lengths (instruction + separator + candidate)")
for kind in ("choice", "score"):
    if lengths.get(kind):
        print(f"  {kind:<7}{len(lengths[kind]):>7}  {percentiles(lengths[kind])}")
