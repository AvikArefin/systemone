"""Baseline-adjusted S1MB scoring.

Implements the three adjustments described in `docs/10-metrics.md`: balanced
accuracy for `noul`, headroom-adjusted accuracy for `choice`, and expected-value
error for `score`. Raw accuracy flatters a model on this benchmark because the
label marginals are heavily skewed (71.5% of `noul` labels are false, the most
common `score` level holds 50% of decisions), so the adjusted number is the only
one comparable to the published leaderboard.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import torch
from torch import Tensor

KINDS = ("noul", "choice", "score")


@dataclass
class Decision:
    benchmark: str
    kind: str
    probabilities: list[float]
    distribution: tuple[float, ...] | None
    label: float
    candidate_ids: tuple[str, ...] | None = None
    values: tuple[float, ...] | None = None


@dataclass
class BenchmarkScore:
    task: str
    kind: str
    decisions: int
    adjusted: float
    raw: float
    baseline: float | None = None


@dataclass
class Report:
    benchmarks: list[BenchmarkScore] = field(default_factory=list)
    task_average: float = 0.0
    task_scores: dict[str, float] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)


def balanced_accuracy(decisions: list[Decision]) -> float | None:
    true_positive = false_positive = false_negative = true_negative = 0.0
    for decision in decisions:
        target = (
            decision.distribution[0]
            if decision.distribution is not None
            else decision.label
        )
        predicted = decision.probabilities[1] >= 0.5 if len(decision.probabilities) > 1 else decision.probabilities[0] >= 0.5
        if predicted:
            true_positive += target
            false_positive += 1.0 - target
        else:
            false_negative += target
            true_negative += 1.0 - target
    positive_mass = true_positive + false_negative
    negative_mass = true_negative + false_positive
    if positive_mass <= 0.0 or negative_mass <= 0.0:
        return None
    recall_true = true_positive / positive_mass
    recall_false = true_negative / negative_mass
    return 2.0 * ((recall_true + recall_false) / 2.0) - 1.0


def _choice_baseline(decisions: list[Decision]) -> float | None:
    """Best trivial policy: uniform, always-one-id, or always-one-position."""
    if not decisions:
        return None
    uniform = sum(1.0 / max(1, len(d.probabilities)) for d in decisions) / len(decisions)
    by_id: dict[str, float] = defaultdict(float)
    by_position: dict[int, float] = defaultdict(float)
    for decision in decisions:
        target = decision.distribution or tuple(
            1.0 if i == int(decision.label) else 0.0
            for i in range(len(decision.probabilities))
        )
        for position, weight in enumerate(target):
            by_position[position] += weight
            if decision.candidate_ids is not None and position < len(decision.candidate_ids):
                by_id[decision.candidate_ids[position]] += weight
    best_position = max(by_position.values()) / len(decisions)
    best_id = max(by_id.values()) / len(decisions) if by_id else 0.0
    return max(uniform, best_position, best_id)


def choice_adjusted(decisions: list[Decision]) -> float | None:
    baseline = _choice_baseline(decisions)
    if baseline is None or baseline >= 1.0:
        return None
    accuracy = sum(
        _probability_at_gold(decision) for decision in decisions
    ) / len(decisions)
    return (accuracy - baseline) / (1.0 - baseline)


def _probability_at_gold(decision: Decision) -> float:
    if decision.distribution is not None:
        return sum(
            weight * decision.probabilities[position]
            for position, weight in enumerate(decision.distribution)
        )
    return decision.probabilities[int(decision.label)]


def _scale(values: tuple[float, ...] | None, count: int) -> list[float]:
    if values is None:
        return [i / (count - 1) if count > 1 else 0.0 for i in range(count)]
    low, high = min(values), max(values)
    if high <= low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def score_adjusted(decisions: list[Decision]) -> float | None:
    if not decisions:
        return None
    errors: list[float] = []
    targets: list[float] = []
    for decision in decisions:
        scale = _scale(decision.values, len(decision.probabilities))
        predicted = sum(p * scale[i] for i, p in enumerate(decision.probabilities))
        weights = decision.distribution
        if weights is None:
            weights = tuple(
                1.0 if i == int(decision.label) else 0.0
                for i in range(len(decision.probabilities))
            )
        target = sum(w * scale[i] for i, w in enumerate(weights))
        errors.append(abs(predicted - target))
        targets.append(target)
    ordered = sorted(targets)
    median = ordered[len(ordered) // 2]
    baseline_mae = sum(abs(t - median) for t in targets) / len(targets)
    if baseline_mae <= 0.0:
        return None
    mae = sum(errors) / len(errors)
    return 1.0 - mae / baseline_mae


ADJUSTMENTS = {
    "noul": balanced_accuracy,
    "choice": choice_adjusted,
    "score": score_adjusted,
}


def score_benchmark(task: str, kind: str, decisions: list[Decision]) -> BenchmarkScore | None:
    if not decisions:
        return None
    adjusted = ADJUSTMENTS[kind](decisions)
    if adjusted is None:
        return None
    raw = (
        sum(_probability_at_gold(d) for d in decisions) / len(decisions)
        if kind in ("choice", "score")
        else sum(
            1.0 if (d.probabilities[1] >= 0.5) == (d.label >= 0.5) else 0.0
            for d in decisions
        )
        / len(decisions)
    )
    baseline = _choice_baseline(decisions) if kind == "choice" else None
    return BenchmarkScore(
        task=task,
        kind=kind,
        decisions=len(decisions),
        adjusted=max(0.0, min(1.0, adjusted)),
        raw=raw,
        baseline=baseline,
    )


def build_report(decisions: list[Decision]) -> Report:
    grouped: dict[tuple[str, str], list[Decision]] = defaultdict(list)
    for decision in decisions:
        grouped[(decision.benchmark, decision.kind)].append(decision)

    report = Report()
    per_task: dict[str, list[float]] = defaultdict(list)
    for (task, kind), group in sorted(grouped.items()):
        scored = score_benchmark(task, kind, group)
        if scored is None:
            report.excluded[kind] = report.excluded.get(kind, 0) + 1
            continue
        report.benchmarks.append(scored)
        per_task[kind].append(scored.adjusted)

    for kind in KINDS:
        report.task_scores[kind] = (
            sum(per_task[kind]) / len(per_task[kind]) if per_task[kind] else 0.0
        )
    if per_task:
        report.task_average = sum(report.task_scores.values()) / len(KINDS)
    return report


def probabilities_by_group(
    model,
    dataset,
    device: torch.device | None = None,
    batch_size: int = 16,
    temperatures: dict[str, float] | None = None,
) -> list[list[float]]:
    """Run the model and return one distribution per dataset example."""
    from .data import EncodedSet

    assert isinstance(dataset, EncodedSet)
    device = device or next(model.parameters()).device
    temperatures = temperatures or {}
    model.to(device).eval()
    out: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            batch = dataset.collate(range(start, stop)).to(device)
            logits = model(
                batch.state_ids, batch.state_mask, batch.query_ids, batch.query_mask
            )
            for group in range(len(batch.kinds)):
                kind = batch.kinds[group]
                index = (batch.group_ids == group).nonzero(as_tuple=True)[0]
                values = logits[index] / temperatures.get(kind, 1.0)
                if kind == "noul":
                    probability = torch.sigmoid(values[0]).item()
                    out.append([1.0 - probability, probability])
                else:
                    out.append(torch.softmax(values, dim=0).tolist())
    return out
