from .schema import Choice, Noul, Score
from .tokenizer import Tokenizer


def _render(tokenizer: Tokenizer, instructions: str, candidate: str | None) -> list[int]:
    ids = tokenizer.encode(instructions)
    if candidate is not None:
        ids = ids + [tokenizer.sep_id] + tokenizer.encode(candidate)
    return ids


def noul_query(tokenizer: Tokenizer, question: Noul) -> list[int]:
    return _render(tokenizer, question.instructions, None)


def choice_query(tokenizer: Tokenizer, question: Choice, option: str) -> list[int]:
    return _render(tokenizer, question.instructions, option)


def score_query(tokenizer: Tokenizer, question: Score, level: str) -> list[int]:
    return _render(tokenizer, question.instructions, level)
