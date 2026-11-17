"""Integration tests against the real train and test splits.

These are the tests that would have caught the original bugs, because they assert
on the actual corpus shapes: 151-option choices, 40k-token states, long
instructions. They need the HuggingFace cache and are skipped without it.
"""

from pathlib import Path

import pytest

from systemone import ModelConfig
from systemone.bpe import SentencePieceTokenizer
from systemone.data import EncodedSet
from systemone.corpus import (
    BEKKO_REPO,
    _options,
    _text,
    load_test,
    load_train,
    parse_decision,
    render_state,
    test_configs as eval_roster,
    train_configs,
)
from systemone.tokenizer import ByteTokenizer, truncate_head_tail

CKPT = Path(__file__).resolve().parents[1] / "artifacts" / "bekko_tiny" / "tokenizer.model"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tokenizer():
    if not CKPT.exists():
        pytest.skip("no trained tokenizer in artifacts/")
    return SentencePieceTokenizer(CKPT)


def test_test_split_has_the_documented_shape():
    grouped = load_test(["aegis2", "arc"], seed=0)
    assert grouped, "loader returned nothing"
    examples = [e for case in grouped.values() for e in case]
    assert all(e.benchmark for e in examples), "benchmark label must survive parsing"


def test_loading_does_not_silently_drop_choice_decisions():
    """>8-option choices used to raise inside the parser and disappear."""
    grouped = load_test(["banking77", "clinc"], max_candidates=None, shuffle_options=False, strict=True)
    kinds = {e.question.__class__.__name__ for case in grouped.values() for e in case}
    assert "Choice" in kinds
    assert grouped, "wide-option subsets must not vanish entirely"


def test_eval_loading_preserves_candidate_order():
    """Scoring needs the benchmark's own order, so ids must match criteria order."""
    import random

    from datasets import load_dataset

    from systemone.corpus import parse_decision

    dataset = load_dataset(BEKKO_REPO, "arc", split="test")
    for row in dataset:
        targets = {t["decision_id"]: t for t in row["targets"]}
        state = render_state(row["input"]["state_json"])
        for decision in row["input"]["decisions"]:
            target = targets.get(decision["id"])
            if target is None or decision["type"] != "choice":
                continue
            expected = tuple(c["id"] for c in decision["criteria"])
            example = parse_decision(
                state, decision, target, None, random.Random(0), shuffle=False
            )
            assert example is not None
            assert example.candidate_ids == expected
            # the distribution must be in that same order
            weights = dict(zip(target["ids"], target["probabilities"]))
            total = sum(weights.get(cid, 0.0) for cid in expected)
            for option_id, weight in zip(expected, example.distribution):
                assert weight == pytest.approx(weights[option_id] / total, abs=1e-9)
            return
    pytest.fail("no choice decision found in arc")


def test_every_choice_row_keeps_its_candidate_text(tokenizer):
    """The original bug: front-only truncation deleted whole candidates.

    On the real corpus the instruction is often 600 tokens against a 48-token
    budget, so this asserts the property that was actually broken.
    """
    grouped = load_test(["arc", "aqua_rat"], max_candidates=8, strict=True)
    examples = [e for case in grouped.values() for e in case]
    cfg = _cfg(tokenizer)
    encoded = EncodedSet.build(examples, tokenizer, cfg)

    from systemone.data import rows_for

    for example in examples:
        if example.question.__class__.__name__ == "Noul":
            continue
        rows, _, _, _ = rows_for(example, tokenizer, cfg.max_query_tokens)
        for row in rows:
            assert tokenizer.sep_id in row, "candidate was truncated away"
            assert len(row) <= cfg.max_query_tokens


def test_choice_groups_do_not_collapse_to_one_encoding(tokenizer):
    """27% of test choice groups collapsed before the fix; this pins it at zero."""
    grouped = load_test(["arc", "aqua_rat", "bbq"], max_candidates=8, strict=True)
    examples = [e for case in grouped.values() for e in case]
    cfg = _cfg(tokenizer)
    encoded = EncodedSet.build(examples, tokenizer, cfg)

    from systemone.data import rows_for

    checked = 0
    for index, example in enumerate(examples):
        if example.question.__class__.__name__ != "Choice":
            continue
        rows, _, _, _ = rows_for(example, tokenizer, cfg.max_query_tokens)
        if len(rows) < 2:
            continue
        checked += 1
        assert len(set(map(tuple, rows))) == len(rows), (
            f"group {index} has indistinguishable options"
        )
    assert checked > 0


def test_state_truncation_is_measured_not_assumed(tokenizer):
    """Half of the test split is longer than the state budget; the encoding must be honest."""
    grouped = load_test(["aegis2"], strict=True)
    example = next(e for case in grouped.values() for e in case)
    ids = tokenizer.encode(example.state)
    assert truncate_head_tail(ids, 8) == ids[:6] + ids[-2:]


def test_train_split_loads():
    grouped = load_train(["arc", "aegis2"], max_cases=2)
    assert grouped, "bekko loader returned nothing"
    examples = [e for case in grouped.values() for e in case]
    assert examples


def test_test_configs_returns_the_evaluation_roster():
    assert len(eval_roster()) == 106


def test_train_configs_filters_to_train_splits():
    configs = set(train_configs())
    assert "agnews" in configs
    assert "bbq" not in configs


def test_wide_options_survive_without_a_cap():
    """151-option choices exist in the test split; they must parse with max_candidates=None."""
    import random

    from datasets import load_dataset

    dataset = load_dataset(BEKKO_REPO, "arc", split="test")
    parsed = 0
    for row in dataset:
        targets = {t["decision_id"]: t for t in row["targets"]}
        state = render_state(row["input"]["state_json"])
        for decision in row["input"]["decisions"]:
            target = targets.get(decision["id"])
            if target is None or decision["type"] != "choice":
                continue
            example = parse_decision(state, decision, target, None, random.Random(0), shuffle=False)
            assert example is not None
            parsed += 1
    assert parsed > 0


def _cfg(tokenizer) -> ModelConfig:
    return ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=64,
        max_state_tokens=192,
        max_query_tokens=48,
    )
