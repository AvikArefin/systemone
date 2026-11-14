import torch

from .model import TinyJev
from .prompt import choice_query, noul_query, score_query
from .schema import (
    Answer,
    Choice,
    Noul,
    Request,
    make_choice,
    make_noul,
    make_score,
)
from .tokenizer import Tokenizer, truncate_head_tail


def _pad(
    sequences: list[list[int]], width: int, pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool, device=device)
    for row, seq in enumerate(sequences):
        if not seq:
            raise ValueError("empty query sequence")
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[row, : len(seq)] = True
    return ids, mask


@torch.no_grad()
def answer_request(
    model: TinyJev,
    tokenizer: Tokenizer,
    request: Request,
    device: torch.device | None = None,
    temperatures: dict[str, float] | None = None,
) -> dict[str, Answer]:
    cfg = model.cfg
    device = device or next(model.parameters()).device
    temperatures = temperatures or {}
    state = truncate_head_tail(tokenizer.encode(request.state), cfg.max_state_tokens)
    if not state:
        raise ValueError("encoded state is empty")

    jobs: list[tuple[str, list[list[int]]]] = []
    limit = cfg.max_query_tokens
    for name, question in request.questions.items():
        if isinstance(question, Noul):
            jobs.append((name, [noul_query(tokenizer, question, limit)]))
        elif isinstance(question, Choice):
            queries = [
                choice_query(tokenizer, question, o, limit) for o in question.criteria
            ]
            jobs.append((name, queries))
        else:
            queries = [
                score_query(tokenizer, question, level, limit)
                for level in question.criteria
            ]
            jobs.append((name, queries))

    flat = [(name, query) for name, queries in jobs for query in queries]
    queries = [query for _, query in flat]
    width = max(len(query) for query in queries)
    query_ids, query_mask = _pad(queries, width, tokenizer.pad_id, device)
    state_ids = torch.tensor([state] * len(queries), dtype=torch.long, device=device)
    state_mask = torch.ones_like(state_ids, dtype=torch.bool)

    logits = model(state_ids, state_mask, query_ids, query_mask)

    grouped: dict[str, list[float]] = {}
    for (name, _), logit in zip(flat, logits.tolist()):
        grouped.setdefault(name, []).append(logit)

    answers: dict[str, Answer] = {}
    for name, question in request.questions.items():
        kind = "noul" if isinstance(question, Noul) else "choice" if isinstance(question, Choice) else "score"
        values = torch.tensor(grouped[name]) / temperatures.get(kind, 1.0)
        if isinstance(question, Noul):
            answers[name] = make_noul(torch.sigmoid(values[0]).item())
        elif isinstance(question, Choice):
            probs = torch.softmax(values, dim=0).tolist()
            answers[name] = make_choice(dict(zip(question.criteria, probs)))
        else:
            probs = torch.softmax(values, dim=0).tolist()
            answers[name] = make_score({i: p for i, p in enumerate(probs)})
    return answers
