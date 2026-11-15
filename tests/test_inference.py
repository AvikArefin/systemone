"""Tests for one-request inference.

`answer_request` must agree with the training-time encoding. It previously had the
same front-truncation bug as `EncodedSet.build`, so a model could be served
differently from how it was trained -- the worst kind of drift, because nothing
crashes.
"""

import pytest

from systemone import ByteTokenizer, Choice, ModelConfig, Noul, Request, Score
from systemone.data import Example, rows_for
from systemone.inference import answer_request
from systemone.model import TinyJev
from systemone.prompt import choice_query


def _cfg(**overrides) -> ModelConfig:
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


def _request(instructions="pick a team", criteria=("billing", "security")):
    return Request(
        state="the customer was charged twice",
        questions={
            "team": Choice(instructions=instructions, criteria=list(criteria)),
            "safe": Noul(instructions="is this safe?"),
            "urgency": Score(instructions="how urgent?", criteria=["low", "high"]),
        },
    )


def test_answer_request_returns_one_answer_per_question(tokenizer):
    model = TinyJev(_cfg()).eval()
    answers = answer_request(model, tokenizer, _request())
    assert set(answers) == {"team", "safe", "urgency"}


def test_choice_probabilities_sum_to_one(tokenizer):
    model = TinyJev(_cfg()).eval()
    answers = answer_request(model, tokenizer, _request())
    assert sum(answers["team"].probabilities.values()) == pytest.approx(1.0)


def test_choice_answer_keys_match_the_criteria(tokenizer):
    criteria = ("billing", "security", "support")
    model = TinyJev(_cfg()).eval()
    answers = answer_request(model, tokenizer, _request(criteria=criteria))
    assert set(answers["team"].probabilities) == set(criteria)
    assert answers["team"].choice in criteria


def test_noul_answer_is_a_probability(tokenizer):
    model = TinyJev(_cfg()).eval()
    answer = answer_request(model, tokenizer, _request())["safe"]
    assert 0.0 <= answer.noul <= 1.0


def test_score_answer_is_an_expected_level(tokenizer):
    model = TinyJev(_cfg()).eval()
    answer = answer_request(model, tokenizer, _request())["urgency"]
    assert 0.0 <= answer.score <= 1.0


def test_temperature_changes_confidence_not_the_winner(tokenizer):
    model = TinyJev(_cfg()).eval()
    plain = answer_request(model, tokenizer, _request())
    warm = answer_request(
        model, tokenizer, _request(), temperatures={"choice": 5.0, "noul": 5.0, "score": 5.0}
    )
    assert plain["team"].choice == warm["team"].choice
    assert warm["team"].confidence <= plain["team"].confidence


def test_long_candidates_are_not_deleted(tokenizer):
    """A candidate longer than the query budget must still reach the model."""
    criteria = ("x" * 500, "y" * 500)
    model = TinyJev(_cfg()).eval()
    answers = answer_request(
        model,
        tokenizer,
        Request(
            state="state",
            questions={"q": Choice(instructions="choose", criteria=list(criteria))},
        ),
    )
    assert sum(answers["q"].probabilities.values()) == pytest.approx(1.0)


def test_inference_encoding_matches_training_encoding(tokenizer):
    """Serving and training must build identical rows for the same example."""
    cfg = _cfg()
    question = Choice(instructions="i" * 400, criteria=["billing", "security"])
    build_rows, _, _, _ = rows_for(
        Example(state="s" * 100, question=question, label="billing"),
        tokenizer,
        cfg.max_query_tokens,
    )
    serve_rows = [
        choice_query(tokenizer, question, option, cfg.max_query_tokens)
        for option in question.criteria
    ]
    assert build_rows == serve_rows


def test_state_is_truncated_to_the_configured_budget(tokenizer):
    cfg = _cfg(max_state_tokens=8)
    model = TinyJev(cfg).eval()
    request = Request(
        state="z" * 500,
        questions={"q": Noul(instructions="is it?")},
    )
    # must not raise or produce a different shape than the model expects
    answer = answer_request(model, tokenizer, request)["q"]
    assert 0.0 <= answer.noul <= 1.0


def test_empty_state_is_rejected(tokenizer):
    model = TinyJev(_cfg()).eval()
    with pytest.raises(ValueError):
        Request(state="", questions={"q": Noul(instructions="is it?")})


def test_single_question_forward_pass_is_batched(tokenizer):
    """All candidates go through one forward call, not one per candidate."""
    model = TinyJev(_cfg()).eval()
    calls = []
    handle = model.register_forward_hook(lambda *args: calls.append(1))
    answer_request(model, tokenizer, _request())
    handle.remove()
    assert len(calls) == 1
