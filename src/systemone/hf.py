import random

from .data import Example
from .schema import Choice, Noul, Score


def _is_unknown(gold: str) -> bool:
    return str(gold).strip().lower() == "unknown"


def _choice_options(row: dict) -> tuple[list[str], str]:
    criteria = [f"{option['id']}: {option['description']}" for option in row["options"]]
    label = next(
        criterion
        for criterion, option in zip(criteria, row["options"])
        if option["id"] == row["gold"]
    )
    return criteria, label


def example_from_row(row: dict) -> Example | None:
    if _is_unknown(row["gold"]):
        return None
    kind = row["type"]
    if kind == "noul":
        question = Noul(instructions=row["instructions"])
        label = str(row["gold"]).lower() == "true"
    elif kind == "choice":
        criteria, label = _choice_options(row)
        question = Choice(instructions=row["instructions"], criteria=criteria)
    elif kind == "score":
        question = Score(instructions=row["instructions"], criteria=list(row["levels"]))
        label = int(row["gold"])
    else:
        raise ValueError(f"unsupported question type: {kind!r}")
    return Example(state=row["state"], question=question, label=label)


def load_jevbench(split: str = "test") -> dict[str, list[Example]]:
    from datasets import load_dataset

    rows = load_dataset("JevBench/jevbench", "questions", split=split)
    grouped: dict[str, list[Example]] = {}
    for row in rows:
        example = example_from_row(row)
        if example is not None:
            grouped.setdefault(row["case_id"], []).append(example)
    return grouped


def split_by_case(
    grouped: dict[str, list[Example]],
    test_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[Example], list[Example]]:
    cases = sorted(grouped)
    random.Random(seed).shuffle(cases)
    cut = max(1, int(len(cases) * (1 - test_fraction)))
    train = [example for case in cases[:cut] for example in grouped[case]]
    test = [example for case in cases[cut:] for example in grouped[case]]
    return train, test
