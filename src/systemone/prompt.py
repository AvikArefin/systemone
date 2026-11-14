from .schema import Choice, Noul, Score
from .tokenizer import Tokenizer, truncate_head_tail

MIN_INSTRUCTION_TOKENS = 4
MIN_CANDIDATE_TOKENS = 1
INSTRUCTION_SHARE = 3  # at most one third of the row is reserved for instructions


def _split_budget(
    total: int, instruction_len: int, candidate_len: int
) -> tuple[int, int]:
    """Budget a row as instructions + separator + candidate.

    The candidate gets priority: it is the only part that differs between the
    rows of one group, so a row that loses its candidate is unanswerable. The
    instructions are identical across a group, so they absorb the truncation --
    but they carry the question itself, so they keep a floor of roughly a third.
    """
    available = total - 1  # one token for the separator
    if available <= 0:
        return 0, 0
    floor = min(instruction_len, max(MIN_INSTRUCTION_TOKENS, available // INSTRUCTION_SHARE))
    cand_budget = min(candidate_len, max(MIN_CANDIDATE_TOKENS, available - floor))
    return available - cand_budget, cand_budget


def _render(
    tokenizer: Tokenizer,
    instructions: str,
    candidate: str | None,
    limit: int,
) -> list[int]:
    instruction_ids = tokenizer.encode(instructions)
    if candidate is None:
        return truncate_head_tail(instruction_ids, limit)
    candidate_ids = tokenizer.encode(candidate)
    instr_budget, cand_budget = _split_budget(
        limit, len(instruction_ids), len(candidate_ids)
    )
    return (
        truncate_head_tail(instruction_ids, instr_budget)
        + [tokenizer.sep_id]
        + truncate_head_tail(candidate_ids, cand_budget)
    )


def noul_query(tokenizer: Tokenizer, question: Noul, limit: int) -> list[int]:
    return _render(tokenizer, question.instructions, None, limit)


def choice_query(
    tokenizer: Tokenizer, question: Choice, option: str, limit: int
) -> list[int]:
    return _render(tokenizer, question.instructions, option, limit)


def score_query(tokenizer: Tokenizer, question: Score, level: str, limit: int) -> list[int]:
    return _render(tokenizer, question.instructions, level, limit)
