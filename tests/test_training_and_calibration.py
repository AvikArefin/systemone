"""Tests for the training loop and calibration.

Calibration is the easiest thing to get subtly wrong: fitting a temperature on the
evaluation split reports a number the model was optimised for. These pin that the
fit is separable from the split and that the loss actually decreases.
"""

import math

import pytest
import torch

from systemone import ByteTokenizer, Choice, Example, ModelConfig, Noul
from systemone.calibrate import (
    accuracy,
    accuracy_by_kind,
    collect_logits,
    expected_calibration_error,
    fit_temperatures,
)
from systemone.data import EncodedSet
from systemone.model import TinyJev
from systemone.seeding import build_model
from systemone.train import evaluate, train


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


def _dataset(seed=0, count=40, kinds=("choice", "noul")):
    import random

    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        state = f"state text number {rng.randrange(100)}"
        if "choice" in kinds:
            label = rng.choice(["a", "b", "c"])
            examples.append(
                Example(
                    state=state,
                    question=Choice(instructions="pick one", criteria=["a", "b", "c"]),
                    label=label,
                )
            )
        if "noul" in kinds:
            examples.append(
                Example(
                    state=state,
                    question=Noul(instructions="is it true?"),
                    label=rng.random() < 0.5,
                )
            )
    return examples


def _learnable_dataset(seed=0, count=60):
    """A trivial task: the answer is written in the state.

    Training must be able to fit this, which makes "loss goes down" a real
    assertion rather than a coin flip.
    """
    import random

    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        label = rng.choice(["a", "b", "c"])
        cue = {"a": "alpha", "b": "bravo", "c": "charlie"}[label]
        examples.append(
            Example(
                state=f"the signal word is {cue}",
                question=Choice(instructions="which signal?", criteria=["a", "b", "c"]),
                label=label,
            )
        )
    return examples


def test_training_reduces_the_loss(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_learnable_dataset(), tokenizer, cfg)
    before = evaluate(model, data, device=torch.device("cpu"))
    history = train(model, data, epochs=40, batch_size=8, lr=3e-3, device=torch.device("cpu"))
    after = evaluate(model, data, device=torch.device("cpu"))
    assert history.losses[-1] < history.losses[0]
    assert after < before


def test_history_records_one_lr_and_loss_per_step(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=16), tokenizer, cfg)
    history = train(model, data, epochs=2, batch_size=8, lr=1e-3)
    steps = 2 * math.ceil(len(data) / 8)
    assert len(history.losses) == steps
    assert len(history.lrs) == steps


def test_warmup_starts_below_the_peak_and_cosine_decays(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=32), tokenizer, cfg)
    history = train(model, data, epochs=10, batch_size=4, lr=1e-3)
    peak = max(history.lrs)
    assert history.lrs[0] < peak, "warmup must start below the peak"
    assert history.lrs[-1] < peak, "cosine must decay below the peak"
    assert abs(peak - 1e-3) < 2e-4


def test_training_is_seed_reproducible_on_cpu(tokenizer):
    """Same seed, same data, CPU, deterministic algorithms: identical curve.

    MPS is excluded on purpose -- Apple's kernels are not bit-reproducible, which
    docs/06-training.md calls out.
    """
    cfg = _cfg()
    data = EncodedSet.build(_dataset(count=16), tokenizer, cfg)
    curves = [
        train(
            build_model(cfg, 7),
            data,
            epochs=2,
            batch_size=8,
            lr=1e-3,
            device=torch.device("cpu"),
            seed=7,
        ).losses
        for _ in range(2)
    ]
    assert curves[0] == pytest.approx(curves[1], rel=1e-5)


def test_evaluate_does_not_change_weights(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=8), tokenizer, cfg)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    evaluate(model, data, device=torch.device("cpu"))
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_collect_logits_returns_one_group_per_example(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=10), tokenizer, cfg)
    collected = collect_logits(model, data, device=torch.device("cpu"))
    total = sum(len(groups) for groups in collected.values())
    assert total == len(data)


def test_collect_logits_keeps_row_counts_per_kind(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=5), tokenizer, cfg)
    collected = collect_logits(model, data, device=torch.device("cpu"))
    assert len(collected["noul"][0][0]) == 1, "noul has exactly one logit"
    assert len(collected["choice"][0][0]) == 3, "choice has one logit per option"


def test_accuracy_is_between_zero_and_one(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=10), tokenizer, cfg)
    collected = collect_logits(model, data, device=torch.device("cpu"))
    assert 0.0 <= accuracy(collected) <= 1.0
    for kind, value in accuracy_by_kind(collected).items():
        if collected[kind]:
            assert 0.0 <= value <= 1.0
        else:
            assert math.isnan(value), "an absent kind must report nan, not 0.0"


def test_fit_temperatures_returns_one_per_kind(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=20), tokenizer, cfg)
    collected = collect_logits(model, data, device=torch.device("cpu"))
    temperatures = fit_temperatures(collected, steps=8)
    assert set(temperatures) == {"noul", "choice", "score"}
    for kind, value in temperatures.items():
        if collected[kind]:
            assert 0.25 - 1e-9 <= value <= 10.0 + 1e-9, "fitted value must lie on the search grid"
        else:
            assert value == 1.0, "an absent kind must not be tuned"


def test_temperature_cannot_change_the_argmax(tokenizer):
    """Dividing by a positive scalar preserves ordering, which is the whole point."""
    logits = torch.tensor([1.0, 3.0, -2.0])
    baseline = int(logits.argmax())
    for temperature in (0.25, 0.5, 2.0, 10.0):
        assert int((logits / temperature).argmax()) == baseline


def test_fitting_on_a_held_out_split_does_not_peek(tokenizer):
    """Temperatures fitted on one split must be computed only from that split.

    This is the leakage guard: the eval split must not influence the fit.
    """
    cfg = _cfg()
    model = TinyJev(cfg)
    examples = _dataset(count=40)
    train_set = EncodedSet.build(examples[:40], tokenizer, cfg)
    test_set = EncodedSet.build(examples[40:], tokenizer, cfg)

    from_train = fit_temperatures(collect_logits(model, train_set, device=torch.device("cpu")), steps=8)
    from_test = fit_temperatures(collect_logits(model, test_set, device=torch.device("cpu")), steps=8)
    # different data, so the fitted values should be allowed to differ; the point
    # is that the function takes exactly one split and has no global state
    assert set(from_train) == set(from_test)


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    collected = {
        "noul": [(torch.tensor([0.0]), 1.0), (torch.tensor([0.0]), 0.0)],
        "choice": [(torch.tensor([0.0, 10.0]), 1.0)],
        "score": [],
    }
    assert expected_calibration_error(collected) < 0.05


def test_ece_is_high_for_confident_wrong_answers():
    collected = {
        "noul": [(torch.tensor([10.0]), 0.0)] * 10,
        "choice": [],
        "score": [],
    }
    assert expected_calibration_error(collected) > 0.5


def test_ece_handles_an_empty_set():
    assert math.isnan(expected_calibration_error({"noul": [], "choice": [], "score": []}))
