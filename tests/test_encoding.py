"""Tests that a row can actually answer the question it encodes.

The original bug: rows were truncated front-only to `max_query_tokens`, so a
long instruction consumed the whole budget and the candidate was deleted. 45% of
S1MB candidate rows lost their option text and 27% of `choice` groups collapsed
to a single identical encoding -- a group the model cannot answer.
"""

import pytest

from systemone import Choice, Example, Noul, Score
from systemone.data import EncodedSet, rows_for
from systemone.prompt import _split_budget, choice_query, noul_query, score_query


def test_split_budget_always_leaves_room_for_the_candidate():
    for total in (8, 16, 48, 192):
        instr_budget, cand_budget = _split_budget(total, 10_000, 10_000)
        assert cand_budget >= 1, f"limit={total} left no candidate budget"
        assert instr_budget + cand_budget + 1 <= total


def test_split_budget_keeps_whole_short_candidate():
    # a short candidate must never be truncated in favour of instructions
    instr_budget, cand_budget = _split_budget(48, 10_000, 10)
    assert cand_budget == 10


def test_split_budget_gives_instructions_a_floor():
    # very long both sides: instructions must still retain some of the question
    instr_budget, _ = _split_budget(48, 10_000, 10_000)
    assert instr_budget >= 4


def test_no_row_ever_loses_its_candidate(tokenizer):
    instruction = "x" * 2_000
    for option in ("short", "y" * 2_000):
        row = choice_query(tokenizer, Choice(instructions=instruction, criteria=["a", "b"]), option, 48)
        assert tokenizer.sep_id in row, "separator missing: candidate was truncated away"
        assert len(row) <= 48


def test_choice_options_stay_distinguishable(tokenizer):
    instruction = "Pick the correct department. " * 40
    criteria = [f"department {i}" for i in range(8)]
    question = Choice(instructions=instruction, criteria=criteria)
    rows = [choice_query(tokenizer, question, option, 48) for option in criteria]
    assert len(set(map(tuple, rows))) == len(rows), "two options encoded identically"


def test_noul_keeps_the_definition_text(tokenizer):
    row = noul_query(tokenizer, Noul(instructions="Is this unsafe?\ntrue: it is\nfalse: it is not"), 64)
    decoded = tokenizer.decode(row)
    assert "unsafe" in decoded


def test_score_levels_stay_distinguishable(tokenizer):
    question = Score(instructions="How urgent? " * 40, criteria=["low", "medium", "high"])
    rows = [score_query(tokenizer, question, level, 48) for level in question.criteria]
    assert len(set(map(tuple, rows))) == 3


def test_rows_for_marks_candidate_kind(tokenizer, cfg):
    example = Example(
        state="state",
        question=Choice(instructions="i" * 200, criteria=["a", "b", "c"]),
        label="b",
    )
    rows, target, kind, _ = rows_for(example, tokenizer, cfg.max_query_tokens)
    assert kind == "choice"
    assert len(rows) == 3
    assert target == 1.0
    assert all(tokenizer.sep_id in row for row in rows)


def test_encoded_set_rejects_a_group_that_lost_its_candidates(tokenizer, monkeypatch):
    """A group whose rows all share one encoding is unanswerable -- fail loudly.

    The guard is what turns a silent accuracy cap into a build error. The budget
    function always protects one candidate token, so this is tested by forcing a
    degenerate encoding rather than by shrinking the limit.
    """
    import systemone.data as data

    example = Example(
        state="state",
        question=Choice(instructions="long instruction here", criteria=["a", "b"]),
        label="a",
    )
    monkeypatch.setattr(
        data,
        "rows_for",
        lambda *args, **kwargs: ([[10, 11], [10, 11]], 0.0, "choice", None),
    )
    with pytest.raises(ValueError, match="candidate text was truncated away"):
        EncodedSet.build([example], tokenizer, _tiny_cfg())


def test_budget_always_preserves_a_candidate_token(tokenizer):
    """Even at the smallest usable limit, one candidate token survives."""
    for limit in (2, 3, 4):
        _, cand_budget = _split_budget(limit, 10_000, 10_000)
        assert cand_budget >= 1
        row = choice_query(
            tokenizer,
            Choice(instructions="i" * 500, criteria=["a", "b"]),
            "b" * 500,
            limit,
        )
        assert tokenizer.sep_id in row and len(row) <= limit


def test_encoded_set_accepts_wide_option_sets(tokenizer):
    """S1MB has choices with up to 151 options; the loader must not choke."""
    cfg = _tiny_cfg()
    criteria = [f"option {i}" for i in range(24)]
    example = Example(
        state="state", question=Choice(instructions="pick one", criteria=criteria), label=criteria[5]
    )
    encoded = EncodedSet.build([example], tokenizer, cfg)
    start, end = int(encoded.row_offsets[0]), int(encoded.row_offsets[1])
    signatures = {
        tuple(encoded.query_data[int(encoded.query_offsets[r]) : int(encoded.query_offsets[r + 1])].tolist())
        for r in range(start, end)
    }
    assert len(signatures) == 24, "wide option sets collapsed to fewer distinct rows"


def test_inference_and_build_agree_on_encoding(tokenizer, cfg):
    """answer_request must encode rows exactly as training does."""
    from systemone import Request, answer_request
    from systemone.model import TinyJev

    criteria = ["billing", "security"]
    request = Request(
        state="a" * 40,
        questions={"q": Choice(instructions="i" * 300, criteria=criteria)},
    )
    model = TinyJev(cfg).eval()
    build_rows, _, _, _ = rows_for(
        Example(state="a" * 40, question=request.questions["q"], label=criteria[0]),
        tokenizer,
        cfg.max_query_tokens,
    )
    inference_rows = [
        choice_query(tokenizer, request.questions["q"], option, cfg.max_query_tokens)
        for option in criteria
    ]
    assert build_rows == inference_rows


def _tiny_cfg(**overrides):
    from systemone import ByteTokenizer, ModelConfig

    defaults = dict(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_state_tokens=32,
        max_query_tokens=24,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)
