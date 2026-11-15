import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .data import EncodedSet
from .model import TinyJev

KIND_ORDER = ("noul", "choice", "score")


def collect_logits(
    model: TinyJev,
    dataset: EncodedSet,
    device: torch.device | None = None,
    batch_size: int = 16,
) -> dict[str, list[tuple[Tensor, float]]]:
    device = device or next(model.parameters()).device
    model.to(device).eval()
    collected: dict[str, list[tuple[Tensor, float]]] = {kind: [] for kind in KIND_ORDER}
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = dataset.collate(
                range(start, min(start + batch_size, len(dataset)))
            ).to(device)
            logits = model(batch.state_ids, batch.state_mask, batch.query_ids, batch.query_mask)
            for group in range(len(batch.kinds)):
                index = (batch.group_ids == group).nonzero(as_tuple=True)[0]
                collected[batch.kinds[group]].append(
                    (logits[index].cpu(), batch.targets[group])
                )
    return collected


def _nll(groups: list[tuple[Tensor, float]], temperature: float, kind: str) -> float:
    total = 0.0
    for logits, target in groups:
        if kind == "noul":
            total += F.binary_cross_entropy_with_logits(
                logits[0] / temperature, torch.tensor(target)
            ).item()
        else:
            total += F.cross_entropy(
                logits / temperature, torch.tensor(int(target))
            ).item()
    return total


def _nll_by_temperature(
    groups: list[tuple[Tensor, float]], kind: str, grid: list[float]
) -> list[float]:
    return [_nll(groups, temperature, kind) for temperature in grid]


def is_identifiable(
    groups: list[tuple[Tensor, float]],
    kind: str,
    grid: list[float],
    tolerance: float = 1e-3,
) -> bool:
    """Can this split actually pin down a temperature?

    When a model is close to random, widening T barely moves the loss, so many
    temperatures fit equally well and the "best" one is just whichever grid point
    the search lands on. Reporting it as a fitted value would be false precision.
    """
    if len(grid) < 2:
        return False
    values = _nll_by_temperature(groups, kind, grid)
    best = min(values)
    # how much the loss changes across the whole grid, relative to its floor
    if best <= 0.0:
        return max(values) - best > tolerance
    return (max(values) - best) / best > tolerance


def fit_temperatures(
    collected: dict[str, list[tuple[Tensor, float]]],
    low: float = 0.25,
    high: float = 10.0,
    steps: int = 40,
    warn: bool = True,
) -> dict[str, float]:
    grid = [math.exp(math.log(low) + i * (math.log(high) - math.log(low)) / (steps - 1)) for i in range(steps)]
    temperatures = {}
    for kind in KIND_ORDER:
        groups = collected.get(kind, [])
        if not groups:
            temperatures[kind] = 1.0
            continue
        best_t, best_nll = 1.0, float("inf")
        for temperature in grid:
            value = _nll(groups, temperature, kind)
            if value < best_nll:
                best_t, best_nll = temperature, value
        temperatures[kind] = best_t
        if warn:
            _warn_if_unidentifiable(groups, kind, grid, best_t, low, high)
    return temperatures


def _warn_if_unidentifiable(
    groups: list[tuple[Tensor, float]],
    kind: str,
    grid: list[float],
    best_t: float,
    low: float,
    high: float,
) -> None:
    if best_t <= grid[0] * (1 + 1e-9) or best_t >= grid[-1] * (1 - 1e-9):
        print(
            f"warning: {kind} temperature hit the grid edge ({best_t:.3g}); "
            f"the optimum is outside [{low:g}, {high:g}]"
        )
    if not is_identifiable(groups, kind, grid):
        print(
            f"warning: {kind} temperature is not identifiable on this split "
            f"({len(groups)} groups); the loss barely changes with T, so "
            f"{best_t:.3g} is not a meaningful fit"
        )


def _confidence(collected: dict[str, list[tuple[Tensor, float]]], temperatures: dict[str, float]) -> tuple[list[float], list[float]]:
    confidences: list[float] = []
    correctness: list[float] = []
    for kind in KIND_ORDER:
        temperature = temperatures.get(kind, 1.0)
        for logits, target in collected.get(kind, []):
            if kind == "noul":
                probability = torch.sigmoid(logits[0] / temperature).item()
                confidence = probability if probability >= 0.5 else 1.0 - probability
                correct = float((probability >= 0.5) == bool(target))
            else:
                probabilities = torch.softmax(logits / temperature, dim=0)
                confidence = probabilities.max().item()
                correct = float(int(probabilities.argmax()) == int(target))
            confidences.append(confidence)
            correctness.append(correct)
    return confidences, correctness


def expected_calibration_error(
    collected: dict[str, list[tuple[Tensor, float]]],
    temperatures: dict[str, float] | None = None,
    bins: int = 10,
) -> float:
    confidences, correctness = _confidence(collected, temperatures or {})
    if not confidences:
        return float("nan")
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            i for i, c in enumerate(confidences)
            if low <= c < high or (index == bins - 1 and c == 1.0)
        ]
        if not members:
            continue
        average_confidence = sum(confidences[i] for i in members) / len(members)
        average_accuracy = sum(correctness[i] for i in members) / len(members)
        error += (len(members) / total) * abs(average_accuracy - average_confidence)
    return error


def accuracy(collected: dict[str, list[tuple[Tensor, float]]]) -> float:
    correct = 0
    total = 0
    for logits, target in collected.get("noul", []):
        correct += int((logits[0].item() >= 0.0) == bool(target))
        total += 1
    for kind in ("choice", "score"):
        for logits, target in collected.get(kind, []):
            correct += int(int(logits.argmax()) == int(target))
            total += 1
    return correct / total if total else float("nan")


def accuracy_by_kind(
    collected: dict[str, list[tuple[Tensor, float]]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for kind in KIND_ORDER:
        correct = 0
        total = 0
        for logits, target in collected.get(kind, []):
            if kind == "noul":
                correct += int((logits[0].item() >= 0.0) == bool(target))
            else:
                correct += int(int(logits.argmax()) == int(target))
            total += 1
        scores[kind] = correct / total if total else float("nan")
    return scores
