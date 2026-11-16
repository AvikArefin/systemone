import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .config import ModelConfig
from .prompt import choice_query, noul_query, score_query
from .schema import Choice, Noul, Question, Score
from .tokenizer import Tokenizer, truncate_head_tail


@dataclass(frozen=True)
class Example:
    state: str
    question: Question
    label: bool | str | int
    distribution: tuple[float, ...] | None = None
    benchmark: str | None = None
    candidate_ids: tuple[str, ...] | None = None
    values: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("example requires state")
        question = self.question
        if isinstance(question, Noul):
            if not isinstance(self.label, bool):
                raise ValueError("noul label must be a bool")
            expected = 1
        elif isinstance(question, Choice):
            if self.label not in question.criteria:
                raise ValueError("choice label must be one of the criteria")
            expected = len(question.criteria)
        else:
            if not (isinstance(self.label, int) and 0 <= self.label < len(question.criteria)):
                raise ValueError("score label must be a level index")
            expected = len(question.criteria)
        if self.distribution is not None:
            if isinstance(question, Noul):
                if len(self.distribution) != 1 or not 0.0 <= self.distribution[0] <= 1.0:
                    raise ValueError("noul distribution must be a single probability")
            else:
                if len(self.distribution) != expected:
                    raise ValueError("distribution length must match the target count")
                if any(p < 0.0 for p in self.distribution):
                    raise ValueError("distribution weights must be non-negative")
                if abs(sum(self.distribution) - 1.0) > 1e-3:
                    raise ValueError("distribution weights must sum to 1")


def question_from_dict(data: dict) -> Question:
    kind = data.get("type")
    if kind == "noul":
        return Noul(instructions=data["instructions"])
    criteria = list(data["criteria"])
    if kind == "choice":
        return Choice(instructions=data["instructions"], criteria=criteria)
    if kind == "score":
        return Score(instructions=data["instructions"], criteria=criteria)
    raise ValueError(f"unknown question type: {kind!r}")


def example_from_dict(data: dict) -> Example:
    return Example(
        state=data["state"],
        question=question_from_dict(data["question"]),
        label=data["label"],
    )


def load_jsonl(path: str | Path) -> list[Example]:
    lines = Path(path).read_text().splitlines()
    return [example_from_dict(json.loads(line)) for line in lines if line.strip()]


@dataclass
class Batch:
    state_ids: Tensor
    state_mask: Tensor
    query_ids: Tensor
    query_mask: Tensor
    group_ids: Tensor
    targets: list[float]
    kinds: list[str]
    distributions: list[tuple[float, ...] | None]

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            self.state_ids.to(device),
            self.state_mask.to(device),
            self.query_ids.to(device),
            self.query_mask.to(device),
            self.group_ids.to(device),
            self.targets,
            self.kinds,
            self.distributions,
        )


def rows_for(
    example: Example, tokenizer: Tokenizer, max_query_tokens: int
) -> tuple[list[list[int]], float, str, tuple[float, ...] | None]:
    question = example.question
    if isinstance(question, Noul):
        return (
            [noul_query(tokenizer, question, max_query_tokens)],
            1.0 if example.label else 0.0,
            "noul",
            example.distribution,
        )
    if isinstance(question, Choice):
        return (
            [
                choice_query(tokenizer, question, option, max_query_tokens)
                for option in question.criteria
            ],
            float(question.criteria.index(example.label)),
            "choice",
            example.distribution,
        )
    return (
        [
            score_query(tokenizer, question, level, max_query_tokens)
            for level in question.criteria
        ],
        float(example.label),
        "score",
        example.distribution,
    )


def _flatten(sequences: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.int64)
    offsets = torch.zeros(len(sequences) + 1, dtype=torch.int64)
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    flat = torch.tensor(
        list(itertools.chain.from_iterable(sequences)), dtype=torch.int32
    )
    return flat, offsets


def _fingerprint(tokenizer: Tokenizer) -> str:
    digest = hashlib.sha256()
    model_path = getattr(tokenizer, "model_path", None)
    if model_path and Path(model_path).exists():
        digest.update(Path(model_path).read_bytes())
    else:
        probe = "the quick brown fox jumps over the lazy dog 0123456789 \u241f"
        digest.update(str(tokenizer.vocab_size).encode())
        digest.update(repr(tokenizer.encode(probe * 8)).encode())
    return digest.hexdigest()[:16]


CACHE_VERSION = 2


@dataclass
class EncodedSet:
    state_data: Tensor
    state_offsets: Tensor
    query_data: Tensor
    query_offsets: Tensor
    row_offsets: Tensor
    targets: Tensor
    kinds: list[str]
    distributions: list[tuple[float, ...] | None]
    pad_id: int
    vocab_size: int
    max_state_tokens: int
    max_query_tokens: int
    fingerprint: str = ""
    benchmarks: list[str | None] | None = None
    candidate_ids: list[tuple[str, ...] | None] | None = None
    values: list[tuple[float, ...] | None] | None = None
    version: int = CACHE_VERSION

    def __len__(self) -> int:
        return self.targets.numel()

    def row_counts(self) -> list[int]:
        return (self.row_offsets[1:] - self.row_offsets[:-1]).tolist()

    def batches(self, max_rows: int = 256) -> list[list[int]]:
        """Group example indices so each batch stays under `max_rows` rows.

        A batch holds examples, and an example expands into one row per
        candidate -- an S1MB choice can carry thousands. Bounding the example
        count therefore bounds nothing; rows are what the attention cost grows
        with, so rows is the number that has to be capped.
        """
        counts = self.row_counts()
        out: list[list[int]] = []
        batch: list[int] = []
        total = 0
        for index, count in enumerate(counts):
            if batch and total + count > max_rows:
                out.append(batch)
                batch, total = [], 0
            batch.append(index)
            total += count
        if batch:
            out.append(batch)
        return out

    @classmethod
    def build(
        cls, examples: list[Example], tokenizer: Tokenizer, cfg: ModelConfig
    ) -> "EncodedSet":
        states: list[list[int]] = []
        queries: list[list[int]] = []
        row_counts: list[int] = []
        targets: list[float] = []
        kinds: list[str] = []
        distributions: list[tuple[float, ...] | None] = []
        benchmarks: list[str | None] = []
        candidate_ids: list[tuple[str, ...] | None] = []
        values: list[tuple[float, ...] | None] = []

        for example in examples:
            state = truncate_head_tail(
                tokenizer.encode(example.state), cfg.max_state_tokens
            )
            if not state:
                raise ValueError("encoded state is empty")
            rows, target, kind, distribution = rows_for(
                example, tokenizer, cfg.max_query_tokens
            )
            if any(not row for row in rows):
                raise ValueError("encoded query is empty")
            if kind != "noul" and any(tokenizer.sep_id not in row for row in rows):
                raise ValueError(
                    "candidate text was truncated away; a group whose rows all "
                    "share one encoding cannot be answered"
                )
            states.append(state)
            queries.extend(rows)
            row_counts.append(len(rows))
            targets.append(target)
            kinds.append(kind)
            distributions.append(distribution)
            benchmarks.append(example.benchmark)
            candidate_ids.append(example.candidate_ids)
            values.append(example.values)

        state_data, state_offsets = _flatten(states)
        query_data, query_offsets = _flatten(queries)
        row_offsets = torch.zeros(len(row_counts) + 1, dtype=torch.int64)
        torch.cumsum(torch.tensor(row_counts, dtype=torch.int64), dim=0, out=row_offsets[1:])

        return cls(
            state_data=state_data,
            state_offsets=state_offsets,
            query_data=query_data,
            query_offsets=query_offsets,
            row_offsets=row_offsets,
            targets=torch.tensor(targets, dtype=torch.float32),
            kinds=kinds,
            distributions=distributions,
            pad_id=tokenizer.pad_id,
            vocab_size=tokenizer.vocab_size,
            max_state_tokens=cfg.max_state_tokens,
            max_query_tokens=cfg.max_query_tokens,
            fingerprint=_fingerprint(tokenizer),
            benchmarks=benchmarks,
            candidate_ids=candidate_ids,
            values=values,
        )

    def compatible(self, tokenizer: Tokenizer, cfg: ModelConfig) -> bool:
        # read from __dict__, not getattr: a cache pickled before `version`
        # existed would otherwise inherit the class default and pass the check
        return (
            self.__dict__.get("version", 0) == CACHE_VERSION
            and self.vocab_size == tokenizer.vocab_size
            and self.pad_id == tokenizer.pad_id
            and self.max_state_tokens == cfg.max_state_tokens
            and self.max_query_tokens == cfg.max_query_tokens
            and getattr(self, "fingerprint", "") == _fingerprint(tokenizer)
        )

    def save(self, path: str | Path) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "EncodedSet":
        return torch.load(path, weights_only=False)

    def collate(self, indices: Sequence[int]) -> Batch:
        index = torch.as_tensor(list(indices), dtype=torch.long)
        state_start = self.state_offsets[index]
        state_end = self.state_offsets[index + 1]
        states = [
            self.state_data[int(start) : int(end)]
            for start, end in zip(state_start, state_end)
        ]

        row_start = self.row_offsets[index]
        row_end = self.row_offsets[index + 1]
        rows: list[int] = []
        for start, end in zip(row_start.tolist(), row_end.tolist()):
            rows.extend(range(start, end))
        row_index = torch.as_tensor(rows, dtype=torch.long)
        query_start = self.query_offsets[row_index]
        query_end = self.query_offsets[row_index + 1]
        queries = [
            self.query_data[int(start) : int(end)]
            for start, end in zip(query_start, query_end)
        ]

        state_ids = pad_sequence(states, batch_first=True, padding_value=self.pad_id)
        state_mask = state_ids != self.pad_id
        query_ids = pad_sequence(queries, batch_first=True, padding_value=self.pad_id)
        query_mask = query_ids != self.pad_id
        group_ids = torch.repeat_interleave(
            torch.arange(len(states)), row_end - row_start
        )

        chosen = list(indices)
        return Batch(
            state_ids=state_ids,
            state_mask=state_mask,
            query_ids=query_ids,
            query_mask=query_mask,
            group_ids=group_ids,
            targets=self.targets[index].tolist(),
            kinds=[self.kinds[i] for i in chosen],
            distributions=[self.distributions[i] for i in chosen],
        )


def fits_budget(example: Example, tokenizer: Tokenizer, cfg: ModelConfig) -> bool:
    """Would this example reach the model whole, with nothing truncated?

    Used to measure (and optionally to train on) only the part of a benchmark
    that fits the budget. It is deliberately strict: a single over-long
    candidate disqualifies the decision.
    """
    if len(tokenizer.encode(example.state)) > cfg.max_state_tokens:
        return False
    question = example.question
    instructions = len(tokenizer.encode(question.instructions))
    criteria = getattr(question, "criteria", None)
    if not criteria:
        return instructions <= cfg.max_query_tokens
    return all(
        instructions + 1 + len(tokenizer.encode(criterion)) <= cfg.max_query_tokens
        for criterion in criteria
    )


def filter_to_budget(
    examples: list[Example], tokenizer: Tokenizer, cfg: ModelConfig
) -> list[Example]:
    return [e for e in examples if fits_budget(e, tokenizer, cfg)]


def load_or_build(
    path: str | Path,
    examples: list[Example],
    tokenizer: Tokenizer,
    cfg: ModelConfig,
) -> EncodedSet:
    path = Path(path)
    if path.exists():
        dataset = EncodedSet.load(path)
        if dataset.compatible(tokenizer, cfg) and len(dataset) == len(examples):
            return dataset
    dataset = EncodedSet.build(examples, tokenizer, cfg)
    dataset.save(path)
    return dataset


def batch_loss(logits: Tensor, batch: Batch) -> Tensor:
    total = torch.zeros((), device=logits.device)
    groups = sorted(set(batch.group_ids.tolist()))
    for group in groups:
        index = (batch.group_ids == group).nonzero(as_tuple=True)[0]
        distribution = batch.distributions[group]
        if batch.kinds[group] == "noul":
            target = torch.tensor(
                distribution[0] if distribution is not None else batch.targets[group],
                device=logits.device,
            )
            total = total + F.binary_cross_entropy_with_logits(logits[index[0]], target)
        elif distribution is not None:
            weights = torch.tensor(distribution, device=logits.device)
            log_probs = F.log_softmax(logits[index], dim=0)
            total = total - (weights * log_probs).sum()
        else:
            target = torch.tensor(int(batch.targets[group]), device=logits.device)
            total = total + F.cross_entropy(logits[index], target)
    return total / len(groups)
