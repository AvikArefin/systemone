from dataclasses import dataclass

MAX_CHOICES = 256
MIN_CHOICES = 2
MAX_LEVELS = 256
MIN_LEVELS = 2


@dataclass(frozen=True)
class Noul:
    instructions: str

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ValueError("noul question requires instructions")


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: list[str]

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ValueError("choice question requires instructions")
        if not MIN_CHOICES <= len(self.criteria) <= MAX_CHOICES:
            raise ValueError(f"choice requires {MIN_CHOICES}..{MAX_CHOICES} options")
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("choice options must be unique")


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: list[str]

    def __post_init__(self) -> None:
        if not self.instructions:
            raise ValueError("score question requires instructions")
        if not MIN_LEVELS <= len(self.criteria) <= MAX_LEVELS:
            raise ValueError(f"score requires {MIN_LEVELS}..{MAX_LEVELS} levels")


Question = Noul | Choice | Score


@dataclass(frozen=True)
class Request:
    state: str
    questions: dict[str, Question]

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("request requires state")
        if not self.questions:
            raise ValueError("request requires at least one question")


@dataclass(frozen=True)
class NoulAnswer:
    type: str
    noul: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.noul <= 1.0:
            raise ValueError("noul probability must be in [0, 1]")


@dataclass(frozen=True)
class ChoiceAnswer:
    type: str
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    type: str
    score: float
    probabilities: dict[int, float]
    confidence: float


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


def make_noul(probability: float) -> NoulAnswer:
    return NoulAnswer(type="noul", noul=float(probability))


def make_choice(probabilities: dict[str, float]) -> ChoiceAnswer:
    choice = max(probabilities, key=probabilities.__getitem__)
    return ChoiceAnswer(
        type="choice",
        choice=choice,
        probabilities=probabilities,
        confidence=probabilities[choice],
    )


def normalise_levels(values: list[float]) -> list[float]:
    """Map an ordered scale onto [0, 1] so expected values are comparable."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def make_score(
    probabilities: dict[int, float], values: list[float] | None = None
) -> ScoreAnswer:
    weights = normalise_levels(values) if values is not None else None
    expected = sum(
        (weights[level] if weights is not None else level) * probability
        for level, probability in probabilities.items()
    )
    return ScoreAnswer(
        type="score",
        score=float(expected),
        probabilities=probabilities,
        confidence=max(probabilities.values()),
    )
