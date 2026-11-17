"""Tests for the typed-question schema and dataset parsing.

The schema caps used to reject option sets larger than 8. The test split has choices with up
to 151 options, so the rejection was swallowed by the parser's broad `except` and
those decisions were dropped without a word. These tests pin both halves of that
interaction.
"""

import random

import pytest

from systemone import Choice, Example, Noul, Request, Score
from systemone.schema import make_choice, make_noul, make_score, normalise_levels
from systemone.corpus import (
    _criterion_definitions,
    _normalise,
    _subsample,
    _text,
    collect_texts,
    parse_decision,
    render_state,
)


def test_question_types_require_instructions():
    for building in (lambda: Noul(instructions=""), lambda: Choice(instructions="", criteria=["a", "b"]), lambda: Score(instructions="", criteria=["a", "b"])):
        with pytest.raises(ValueError, match="requires instructions"):
            building()


def test_choice_rejects_bad_option_counts():
    with pytest.raises(ValueError):
        Choice(instructions="q", criteria=["only-one"])
    with pytest.raises(ValueError):
        Choice(instructions="q", criteria=[f"o{i}" for i in range(1000)])


def test_choice_accepts_realistic_wide_option_sets():
    """The test split reaches 151 options; rejecting these silently loses whole subsets."""
    for count in (2, 8, 9, 17, 36, 77, 151):
        criteria = [f"option {i}" for i in range(count)]
        assert len(Choice(instructions="q", criteria=criteria).criteria) == count


def test_choice_rejects_duplicate_options():
    with pytest.raises(ValueError, match="unique"):
        Choice(instructions="q", criteria=["a", "a"])


def test_score_rejects_bad_level_counts():
    with pytest.raises(ValueError):
        Score(instructions="q", criteria=["one"])


def test_example_validates_label_against_the_question():
    question = Choice(instructions="q", criteria=["a", "b"])
    with pytest.raises(ValueError, match="one of the criteria"):
        Example(state="s", question=question, label="nope")


def test_example_validates_distribution_length_and_mass():
    question = Choice(instructions="q", criteria=["a", "b"])
    with pytest.raises(ValueError, match="length"):
        Example(state="s", question=question, label="a", distribution=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="sum to 1"):
        Example(state="s", question=question, label="a", distribution=(0.5, 0.2))
    with pytest.raises(ValueError, match="non-negative"):
        Example(state="s", question=question, label="a", distribution=(-0.5, 1.5))


def test_example_validates_noul_distribution():
    with pytest.raises(ValueError):
        Example(state="s", question=Noul(instructions="q"), label=True, distribution=(1.5,))


def test_request_requires_a_question():
    with pytest.raises(ValueError):
        Request(state="s", questions={})


def test_normalise_levels_maps_to_unit_interval():
    assert normalise_levels([0.0, 1.0, 2.0, 3.0]) == [0.0, 1 / 3, 2 / 3, 1.0]
    assert normalise_levels([5.0]) == [0.0]
    assert normalise_levels([]) == []


def test_make_score_reports_expected_value_on_the_real_scale():
    answer = make_score({0: 0.0, 1: 0.0, 2: 1.0}, values=[0.0, 0.5, 1.0])
    assert answer.score == pytest.approx(1.0)


def test_make_choice_picks_argmax():
    answer = make_choice({"a": 0.2, "b": 0.7, "c": 0.1})
    assert answer.choice == "b"
    assert answer.confidence == pytest.approx(0.7)


def test_make_noul_rejects_out_of_range():
    with pytest.raises(ValueError):
        make_noul(1.5)


# ---------------------------------------------------------------- parsing


def _decision(kind="noul", criteria=None, instructions="is it?"):
    return {
        "id": "d1",
        "type": kind,
        "kind": "judgment",
        "instructions_json": f'"{instructions}"',
        "criteria": criteria
        or [
            {"id": "false", "description_json": '"it is not"', "value": None},
            {"id": "true", "description_json": '"it is"', "value": None},
        ],
        "documents": [],
    }


def test_parse_noul_includes_the_label_definitions():
    """The two labels differ only by definition text; dropping it hides the task."""
    decision = _decision()
    target = {"ids": ["false", "true"], "probabilities": [0.25, 0.75]}
    example = parse_decision("state", decision, target, 8, random.Random(0))
    assert "it is not" in example.question.instructions
    assert "it is" in example.question.instructions
    assert example.distribution == (0.75,)
    assert example.label is True


def test_criterion_definitions_renders_both_labels():
    text = _criterion_definitions(_decision())
    assert text.count("\n") == 1
    assert text.startswith("false:")


def test_parse_choice_carries_candidate_ids():
    criteria = [{"id": c, "description_json": f'"{c} option"'} for c in "ABC"]
    decision = _decision("choice", criteria)
    target = {"ids": ["A", "B", "C"], "probabilities": [0.2, 0.7, 0.1]}
    example = parse_decision(
        "state", decision, target, 8, random.Random(0), shuffle=False
    )
    assert example.candidate_ids == ("A", "B", "C")
    assert example.distribution == pytest.approx((0.2, 0.7, 0.1))


def test_parse_choice_shuffling_keeps_ids_and_weights_aligned():
    """Subsampling reshuffles rows; ids, criteria and distribution must move together."""
    criteria = [{"id": f"o{i}", "description_json": f'"option body {i}"'} for i in range(6)]
    decision = _decision("choice", criteria)
    probabilities = [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]
    target = {"ids": [f"o{i}" for i in range(6)], "probabilities": probabilities}
    for seed in range(20):
        example = parse_decision("state", decision, target, 8, random.Random(seed))
        for option_id, criterion, weight in zip(
            example.candidate_ids, example.question.criteria, example.distribution
        ):
            index = int(option_id[1:])
            assert criterion == f"{option_id}: option body {index}"
            assert weight == pytest.approx(probabilities[index])


def test_parse_choice_keeps_the_gold_option_under_subsampling():
    criteria = [{"id": f"o{i}", "description_json": f'"opt {i}"'} for i in range(40)]
    decision = _decision("choice", criteria)
    target = {"ids": [f"o{i}" for i in range(40)], "probabilities": [0.025] * 40}
    target["probabilities"][37] = 1.0
    for seed in range(5):
        example = parse_decision("state", decision, target, 8, random.Random(seed))
        assert len(example.candidate_ids) == 8
        assert "o37" in example.candidate_ids, "gold option must survive subsampling"


def test_parse_choice_without_a_cap_keeps_every_option():
    criteria = [{"id": f"o{i}", "description_json": f'"opt {i}"'} for i in range(40)]
    decision = _decision("choice", criteria)
    target = {"ids": [f"o{i}" for i in range(40)], "probabilities": [0.025] * 40}
    example = parse_decision("state", decision, target, None, random.Random(0), shuffle=False)
    assert len(example.candidate_ids) == 40
    assert example.candidate_ids[0] == "o0", "order must be preserved for scoring"


def test_parse_choice_without_a_cap_is_not_dropped():
    """This is the regression: >8 options + no cap used to raise and vanish."""
    criteria = [{"id": f"o{i}", "description_json": f'"opt {i}"'} for i in range(20)]
    decision = _decision("choice", criteria)
    target = {"ids": [f"o{i}" for i in range(20)], "probabilities": [0.05] * 20}
    assert parse_decision("state", decision, target, None, random.Random(0)) is not None


def test_parse_score_orders_levels_by_value():
    criteria = [
        {"id": "high", "description_json": '"high"', "value": 2.0},
        {"id": "low", "description_json": '"low"', "value": 0.0},
        {"id": "mid", "description_json": '"mid"', "value": 1.0},
    ]
    decision = _decision("score", criteria)
    target = {"ids": ["low", "mid", "high"], "probabilities": [0.1, 0.2, 0.7]}
    example = parse_decision("state", decision, target, 8, random.Random(0))
    assert example.question.criteria == ["low", "mid", "high"]
    assert example.values == (0.0, 1.0, 2.0)
    assert example.label == 2
    assert example.distribution == pytest.approx((0.1, 0.2, 0.7))


def test_parse_score_accepts_numeric_target_ids():
    criteria = [{"id": str(i), "description_json": f'"{i}"', "value": float(i)} for i in range(5)]
    decision = _decision("score", criteria)
    target = {"ids": ["3"], "probabilities": [1.0]}
    example = parse_decision("state", decision, target, 8, random.Random(0))
    assert example.distribution[3] == pytest.approx(1.0)


def test_parse_returns_none_on_missing_target():
    decision = _decision("choice", [{"id": "A", "description_json": '"a"'}])
    target = {"ids": [], "probabilities": []}
    assert parse_decision("state", decision, target, 8, random.Random(0)) is None


def test_normalise_returns_none_for_zero_mass():
    assert _normalise([0.0, 0.0]) is None
    assert _normalise([1.0, 1.0]) == (0.5, 0.5)


def test_subsample_is_deterministic_for_a_seed():
    options = [(f"o{i}", f"opt {i}") for i in range(20)]
    a = _subsample(options, "o5", 8, random.Random(3))
    b = _subsample(options, "o5", 8, random.Random(3))
    assert a == b


def test_render_state_handles_the_shapes_in_the_corpus():
    assert render_state('"plain text"') == "plain text"
    assert render_state('["a", "b"]') == "a\nb"
    assert render_state('{"only": "single value"}') == "single value"
    rendered = render_state('{"subject": "hi", "body": "there"}')
    assert "subject: hi" in rendered and "body: there" in rendered


def test_text_unquotes_json_strings():
    assert _text('"quoted"') == "quoted"
    assert _text('{"a": 1}') == '{"a": 1}'
    assert _text("bare") == "bare"


def test_collect_texts_keeps_one_copy_of_text_shared_by_options():
    """A choice case repeats its state and instruction once per option.

    That repetition is an artifact of the batch layout, not of the language, and
    it is what let rubric words win whole-piece merges while document words
    stayed split. The tokenizer corpus must contain each distinct text once.
    """
    question = Choice(instructions="rate it", criteria=["one", "two"])
    grouped = {
        "case/1": [
            Example(state="document", question=question, label="one"),
            Example(state="document", question=question, label="two"),
        ]
    }
    assert collect_texts(grouped) == ["document", "rate it", "one", "two"]


def test_collect_texts_can_reproduce_the_repeated_corpus():
    """The historical comparison tokenizers were built from the repeated list."""
    question = Choice(instructions="rate it", criteria=["one", "two"])
    grouped = {
        "case/1": [
            Example(state="document", question=question, label="one"),
            Example(state="document", question=question, label="two"),
        ]
    }
    assert collect_texts(grouped, deduplicate=False) == [
        "document", "rate it", "one", "two",
        "document", "rate it", "one", "two",
    ]
