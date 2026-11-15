"""Tests for baseline-adjusted scoring.

The point of these is that a score of 0 must mean "no better than the trivial
policy" and 1 must mean "perfect", regardless of how skewed the labels are. A
metric that silently reports raw accuracy cannot be compared to the leaderboard.
"""

import pytest

from systemone.scoring import (
    Decision,
    balanced_accuracy,
    build_report,
    choice_adjusted,
    score_adjusted,
)


def noul(labels_and_probs):
    return [
        Decision(
            benchmark="b",
            kind="noul",
            probabilities=[1.0 - p, p],
            distribution=(label,),
            label=label,
        )
        for label, p in labels_and_probs
    ]


def test_balanced_accuracy_is_zero_for_a_constant_answer():
    # always say false, and the labels are 80% false: raw accuracy is high,
    # balanced accuracy is exactly chance
    decisions = noul([(0.0, 0.0)] * 80 + [(1.0, 0.0)] * 20)
    assert balanced_accuracy(decisions) == pytest.approx(0.0, abs=1e-6)


def test_balanced_accuracy_is_zero_for_always_true():
    decisions = noul([(0.0, 1.0)] * 80 + [(1.0, 1.0)] * 20)
    assert balanced_accuracy(decisions) == pytest.approx(0.0, abs=1e-6)


def test_balanced_accuracy_is_one_when_perfect():
    decisions = noul([(0.0, 0.0)] * 10 + [(1.0, 1.0)] * 10)
    assert balanced_accuracy(decisions) == pytest.approx(1.0)


def test_balanced_accuracy_is_negative_when_inverted():
    decisions = noul([(0.0, 1.0)] * 10 + [(1.0, 0.0)] * 10)
    assert balanced_accuracy(decisions) == pytest.approx(-1.0)


def test_balanced_accuracy_uses_soft_targets():
    """A soft '70% true' label contributes to both classes, not to one."""
    decisions = [
        Decision(
            benchmark="b",
            kind="noul",
            probabilities=[0.5, 0.5],
            distribution=(0.7,),
            label=1.0,
        )
    ]
    # a coin-flip prediction: recall_true = 0.7, recall_false = 0.3 -> BA 0.5
    assert balanced_accuracy(decisions) == pytest.approx(0.0, abs=1e-6)


def test_balanced_accuracy_returns_none_for_one_class():
    decisions = noul([(1.0, 1.0)] * 5)
    assert balanced_accuracy(decisions) is None


def choice(gold_and_pred, n_options=4):
    out = []
    for gold, pred in gold_and_pred:
        probs = [0.0] * n_options
        probs[pred] = 1.0
        out.append(
            Decision(
                benchmark="b",
                kind="choice",
                probabilities=probs,
                distribution=tuple(1.0 if i == gold else 0.0 for i in range(n_options)),
                label=float(gold),
                candidate_ids=tuple("ABCD"[:n_options]),
            )
        )
    return out


def test_choice_adjusted_is_zero_for_a_constant_position():
    """Always picking position 0 is the baseline, so it must score exactly 0."""
    decisions = choice([(0, 0)] * 50 + [(1, 0)] * 50)
    assert choice_adjusted(decisions) == pytest.approx(0.0, abs=1e-6)


def test_choice_adjusted_is_one_when_perfect():
    decisions = choice([(i % 4, i % 4) for i in range(100)])
    assert choice_adjusted(decisions) == pytest.approx(1.0, abs=1e-6)


def test_choice_adjusted_rewards_beating_the_baseline():
    # golds uniform over 4 positions, model right 50% of the time
    decisions = choice([(i % 4, i % 4 if i % 2 == 0 else (i + 1) % 4) for i in range(100)])
    value = choice_adjusted(decisions)
    assert 0.2 < value < 0.5


def test_choice_baseline_considers_candidate_ids():
    """A skewed id distribution makes 'always pick that id' the real baseline."""
    out = []
    for i in range(100):
        gold = 0 if i < 70 else i % 4
        probs = [0.0] * 4
        probs[gold] = 1.0
        out.append(
            Decision(
                benchmark="b",
                kind="choice",
                probabilities=probs,
                distribution=tuple(1.0 if j == gold else 0.0 for j in range(4)),
                label=float(gold),
                candidate_ids=("A", "B", "C", "D"),
            )
        )
    # a perfect model scores 1.0 despite the skewed baseline
    assert choice_adjusted(out) == pytest.approx(1.0)


def test_choice_soft_targets_give_partial_credit():
    decisions = [
        Decision(
            benchmark="b",
            kind="choice",
            probabilities=[0.25, 0.25, 0.25, 0.25],
            distribution=(0.25, 0.25, 0.25, 0.25),
            label=0.0,
            candidate_ids=("A", "B", "C", "D"),
        )
    ]
    assert choice_adjusted(decisions) == pytest.approx(0.0, abs=1e-6)


def score(gold_and_pred, n_levels=5):
    out = []
    for gold, pred in gold_and_pred:
        probs = [0.0] * n_levels
        probs[pred] = 1.0
        out.append(
            Decision(
                benchmark="b",
                kind="score",
                probabilities=probs,
                distribution=tuple(1.0 if i == gold else 0.0 for i in range(n_levels)),
                label=float(gold),
                values=tuple(float(i) for i in range(n_levels)),
            )
        )
    return out


def test_score_adjusted_is_zero_for_the_constant_median():
    """Always predicting the middle level matches the best-constant baseline."""
    decisions = score([(i % 5, 2) for i in range(100)])
    assert score_adjusted(decisions) == pytest.approx(0.0, abs=1e-6)


def test_score_adjusted_is_one_when_perfect():
    decisions = score([(i % 5, i % 5) for i in range(50)])
    assert score_adjusted(decisions) == pytest.approx(1.0, abs=1e-6)


def test_score_adjusted_is_negative_for_a_bad_constant():
    # golds span the range, model always says level 0
    decisions = score([(i % 5, 0) for i in range(100)])
    assert score_adjusted(decisions) < 0.0


def test_score_uses_real_values_not_indices():
    """A 0/0.25/0.5/0.75/1 scale must not be scored as the indices 0..4."""
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    decisions = [
        Decision(
            benchmark="b",
            kind="score",
            probabilities=[0.0, 0.0, 1.0, 0.0, 0.0],  # predicts the middle level
            distribution=tuple(1.0 if i == gold else 0.0 for i in range(5)),
            label=float(gold),
            values=levels,
        )
        for gold in [0, 0, 1, 2, 3, 4, 4]
    ]
    # middle prediction has 0.5 expected value; with a normalised scale the
    # error is at most 0.5, whereas index-space error would be up to 2.0
    value = score_adjusted(decisions)
    assert value is not None and value > -1.0


def test_report_clips_per_benchmark_and_averages_tasks():
    decisions = (
        noul([(0.0, 0.0)] * 10 + [(1.0, 1.0)] * 10)
        + choice([(i % 4, i % 4) for i in range(20)])
        + score([(i % 5, i % 5) for i in range(20)])
    )
    report = build_report(decisions)
    assert sorted(b.kind for b in report.benchmarks) == ["choice", "noul", "score"]
    for bench in report.benchmarks:
        assert 0.0 <= bench.adjusted <= 1.0
    assert report.task_average == pytest.approx(1.0)


def test_report_clips_a_catastrophic_benchmark_instead_of_letting_it_dominate():
    decisions = noul([(0.0, 1.0)] * 10 + [(1.0, 0.0)] * 10)  # perfectly inverted
    report = build_report(decisions)
    assert report.benchmarks[0].adjusted == 0.0


def test_report_excludes_a_benchmark_with_no_headroom():
    """If a trivial policy is already perfect, the benchmark cannot discriminate."""
    decisions = choice([(0, 0)] * 20)
    report = build_report(decisions)
    assert report.benchmarks == []
    assert report.excluded.get("choice") == 1


def test_report_groups_by_benchmark_and_kind():
    decisions = noul([(0.0, 0.0)] * 5 + [(1.0, 1.0)] * 5)
    for decision in decisions:
        decision.benchmark = "other"
    decisions += noul([(0.0, 0.0)] * 5 + [(1.0, 1.0)] * 5)
    report = build_report(decisions)
    assert len(report.benchmarks) == 2
    assert {b.task for b in report.benchmarks} == {"b", "other"}
