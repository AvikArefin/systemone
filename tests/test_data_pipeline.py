"""Tests for the flat token cache, collation, and per-group loss.

These catch the class of bug where the cache silently drifts from the examples:
misaligned offsets, mixing up which rows belong to which group, or a target that
points at the wrong candidate.
"""

import torch

from systemone import ByteTokenizer, Choice, Example, Noul, Score
from systemone.data import CACHE_VERSION, EncodedSet, batch_loss, rows_for


def _examples():
    return [
        Example(state="refund request", question=Noul(instructions="is it urgent?"), label=True),
        Example(
            state="billing question",
            question=Choice(instructions="which team?", criteria=["billing", "security", "support"]),
            label="security",
        ),
        Example(
            state="urgency",
            question=Score(instructions="how urgent?", criteria=["low", "mid", "high"]),
            label=2,
        ),
    ]


def test_len_counts_examples_not_rows(tokenizer, cfg):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    assert len(encoded) == 3
    rows = len(encoded.query_offsets) - 1
    assert rows == 1 + 3 + 3


def test_batches_cover_every_example_once_and_respect_the_row_cap(tokenizer, cfg):
    """A batch is capped by rows, not examples -- an example is one row per candidate."""
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    counts = encoded.row_counts()
    assert counts == [1, 3, 3]

    batches = encoded.batches(max_rows=4)
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2]
    for batch in batches:
        assert sum(counts[index] for index in batch) <= 4


def test_one_huge_example_forms_its_own_batch(tokenizer, cfg):
    """A 2,000-candidate choice must not drag other examples into its grid."""
    wide = Example(
        state="pick one",
        question=Choice(instructions="which?", criteria=[f"option {i}" for i in range(12)]),
        label="option 3",
    )
    encoded = EncodedSet.build([wide, *(_examples() * 4)], tokenizer, cfg)
    counts = encoded.row_counts()
    batches = encoded.batches(max_rows=8)
    assert counts[0] == 12
    assert [batch for batch in batches if 0 in batch] == [[0]], "over the cap, so alone"
    for batch in batches:
        if batch != [0]:
            assert sum(counts[index] for index in batch) <= 8


def test_fits_budget_is_strict_about_candidates(tokenizer, cfg):
    """One over-long candidate disqualifies the decision, even if the state fits."""
    from systemone.data import fits_budget

    long_state = Example(
        state="word " * (cfg.max_state_tokens + 5),
        question=Noul(instructions="ok?"),
        label=True,
    )
    assert not fits_budget(long_state, tokenizer, cfg)

    short = Example(state="tiny", question=Noul(instructions="ok?"), label=True)
    assert fits_budget(short, tokenizer, cfg)

    wide = Example(
        state="tiny",
        question=Choice(
            instructions="pick",
            criteria=["yes", "word " * (cfg.max_query_tokens + 5)],
        ),
        label="yes",
    )
    assert not fits_budget(wide, tokenizer, cfg), "a long option kills the row set"
    assert fits_budget(
        Example(state="tiny", question=Choice(instructions="pick", criteria=["a", "b"]), label="a"),
        tokenizer,
        cfg,
    )


def test_offsets_are_monotonic_and_cover_the_flat_buffers(tokenizer, cfg):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    for offsets, data in (
        (encoded.state_offsets, encoded.state_data),
        (encoded.query_offsets, encoded.query_data),
        (encoded.row_offsets, None),
    ):
        values = offsets.tolist()
        assert values[0] == 0
        assert all(b >= a for a, b in zip(values, values[1:])), "offsets must not decrease"
        if data is not None:
            assert values[-1] == data.numel(), "offsets must cover the buffer exactly"


def test_targets_are_parallel_to_examples(tokenizer, cfg):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    assert encoded.kinds == ["noul", "choice", "score"]
    assert encoded.targets.tolist() == [1.0, 1.0, 2.0]


def test_collate_keeps_group_membership(tokenizer, cfg):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    batch = encoded.collate([0, 1, 2])
    assert batch.group_ids.tolist() == [0, 1, 1, 1, 2, 2, 2]
    assert len(batch.kinds) == 3
    assert len(batch.targets) == 3


def test_collate_preserves_row_order_within_a_group(tokenizer, cfg):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    batch = encoded.collate([1])
    # row 0 must be the first criterion, so argmax maps back to criteria order
    rows = [batch.query_ids[i, : int(batch.query_mask[i].sum())].tolist() for i in range(3)]
    assert len(set(map(tuple, rows))) == 3
    decoded = [tokenizer.decode(row) for row in rows]
    assert "billing" in decoded[0] and "security" in decoded[1] and "support" in decoded[2]


def test_collate_of_a_subset_uses_only_those_examples(tokenizer, cfg):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    batch = encoded.collate([2])
    assert batch.group_ids.tolist() == [0, 0, 0]
    assert batch.targets == [2.0]


def test_masks_hide_padding(tokenizer, cfg):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    batch = encoded.collate([0, 1, 2])
    pad = encoded.pad_id
    assert torch.equal(batch.state_mask, batch.state_ids != pad)
    assert torch.equal(batch.query_mask, batch.query_ids != pad)
    assert not batch.state_mask[batch.state_ids == pad].any()


def test_noul_loss_matches_bce_on_soft_target(tokenizer, cfg):
    example = Example(
        state="s", question=Noul(instructions="is it?"), label=True, distribution=(0.75,)
    )
    encoded = EncodedSet.build([example], tokenizer, cfg)
    batch = encoded.collate([0])
    logits = torch.zeros(1)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[0], torch.tensor(0.75))
    assert torch.isclose(batch_loss(logits, batch), expected)


def test_soft_label_loss_uses_the_distribution(tokenizer, cfg):
    example = Example(
        state="s",
        question=Choice(instructions="which?", criteria=["a", "b"]),
        label="a",
        distribution=(0.8, 0.2),
    )
    encoded = EncodedSet.build([example], tokenizer, cfg)
    batch = encoded.collate([0])
    logits = torch.tensor([2.0, -1.0])
    log_probs = torch.log_softmax(logits, dim=0)
    expected = -(0.8 * log_probs[0] + 0.2 * log_probs[1])
    assert torch.isclose(batch_loss(logits, batch), expected)


def test_group_loss_weights_every_question_equally(tokenizer, cfg):
    """A 2-option question and an 8-option question must count the same."""
    small = Example(
        state="s", question=Choice(instructions="q", criteria=["a", "b"]), label="a"
    )
    large = Example(
        state="s",
        question=Choice(instructions="q", criteria=[f"o{i}" for i in range(8)]),
        label="o0",
    )
    encoded = EncodedSet.build([small, large], tokenizer, cfg)
    batch = encoded.collate([0, 1])
    logits = torch.zeros(len(batch.group_ids))
    # both groups are uniform, so each contributes ln(n)/1: mean over 2 groups
    expected = (torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(8.0))) / 2
    assert torch.isclose(batch_loss(logits, batch), expected)


def test_cache_version_is_stamped(tokenizer, cfg):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    assert encoded.version == CACHE_VERSION


def test_cache_without_a_version_is_rejected(tokenizer, cfg, tmp_path):
    """A cache pickled before `version` existed must not be reused.

    Old caches contain the front-truncation rows, so accepting them would silently
    restore the bug this project just fixed.
    """
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    encoded.__dict__.pop("version", None)
    assert not encoded.compatible(tokenizer, cfg)


def test_cache_rejects_a_different_tokenizer(tokenizer, cfg, tmp_path):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)

    class OtherTokenizer(ByteTokenizer):
        def encode(self, text):  # different ids, same vocab size
            return [i + 1 for i in super().encode(text)]

    assert not encoded.compatible(OtherTokenizer(), cfg)


def test_round_trip_through_disk(tokenizer, cfg, tmp_path):
    encoded = EncodedSet.build(_examples(), tokenizer, cfg)
    path = tmp_path / "cache.pt"
    encoded.save(path)
    loaded = EncodedSet.load(path)
    assert loaded.compatible(tokenizer, cfg)
    assert len(loaded) == len(encoded)
    assert loaded.kinds == encoded.kinds


def test_row_offsets_match_one_row_per_criterion(tokenizer, cfg):
    examples = _examples()
    encoded = EncodedSet.build(examples, tokenizer, cfg)
    for index, example in enumerate(examples):
        rows, _, _, _ = rows_for(example, tokenizer, cfg.max_query_tokens)
        span = int(encoded.row_offsets[index + 1] - encoded.row_offsets[index])
        assert span == len(rows)
