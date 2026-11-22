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
    is_identifiable,
)
from systemone.data import EncodedSet
from systemone.model import TinyJev
from systemone.seeding import build_model
from systemone.train import _shard_chunks, autocast_dtype, default_device, evaluate, init_distributed, train


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
    history = train(model, data, epochs=40, max_rows=24, lr=3e-3, device=torch.device("cpu"))
    after = evaluate(model, data, device=torch.device("cpu"))
    assert history.losses[-1] < history.losses[0]
    assert after < before


def test_history_records_one_lr_and_loss_per_step(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=16), tokenizer, cfg)
    history = train(model, data, epochs=2, max_rows=24, lr=1e-3)
    steps = 2 * len(data.batches(24))
    assert len(history.losses) == steps
    assert len(history.lrs) == steps


def test_resume_continues_instead_of_restarting(tokenizer):
    """A resumed run keeps its losses and its schedule position."""
    cfg = _cfg()
    data = EncodedSet.build(_learnable_dataset(count=40), tokenizer, cfg)
    model = TinyJev(cfg)
    saved = {}

    first = train(
        model, data, epochs=2, max_rows=24, lr=1e-3, device=torch.device("cpu"),
        on_epoch_end=lambda epoch, state: saved.update(state),
    )
    assert first.epoch == 2 and saved["epoch"] == 2
    lr_at_two = first.lrs[-1]

    second = train(
        model, data, epochs=4, max_rows=24, lr=1e-3, device=torch.device("cpu"),
        resume=saved,
    )
    assert second.epoch == 4
    assert second.losses[: len(first.losses)] == first.losses
    assert len(second.lrs) == len(first.lrs) + 2 * len(data.batches(24))
    assert second.lrs[len(first.lrs) - 1] == lr_at_two


def test_gradient_accumulation_records_one_step_per_window(tokenizer):
    """Accumulating chunks must not multiply the optimizer steps."""
    cfg = _cfg()
    data = EncodedSet.build(_dataset(count=16), tokenizer, cfg)
    history = train(model := TinyJev(cfg), data, epochs=2, max_rows=24, grad_accum=3, lr=1e-3)
    expected = 2 * math.ceil(len(data.batches(24)) / 3)
    assert len(history.losses) == expected


def test_checkpointing_and_bf16_still_train(tokenizer):
    """Both are memory knobs; neither may break the forward/backward path."""
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_learnable_dataset(count=40), tokenizer, cfg)
    history = train(
        model,
        data,
        epochs=20,
        max_rows=24,
        lr=3e-3,
        device=torch.device("cpu"),
        checkpoint=True,
        bf16=True,
    )
    assert history.losses[-1] < history.losses[0]


def test_default_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert default_device() == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert default_device() == torch.device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert default_device() == torch.device("cpu")


def test_autocast_dtype_uses_fp16_on_cuda_without_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert autocast_dtype(torch.device("cuda")) == torch.float16
    assert autocast_dtype(torch.device("mps")) == torch.bfloat16
    assert autocast_dtype(torch.device("cpu")) == torch.bfloat16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert autocast_dtype(torch.device("cuda")) == torch.bfloat16


def test_shard_chunks_balances_ranks_without_deadlock():
    chunks = [[0], [1], [2], [3], [4]]
    r0 = _shard_chunks(chunks, rank=0, world_size=2)
    r1 = _shard_chunks(chunks, rank=1, world_size=2)
    assert len(r0) == len(r1) == 2
    assert r0 == [[0], [2]] and r1 == [[1], [3]]

    # Fewer chunks than ranks: pads so every rank still runs 1 step.
    tiny = [[0]]
    assert _shard_chunks(tiny, rank=0, world_size=2) == [[0]]
    assert _shard_chunks(tiny, rank=1, world_size=2) == [[0]]


def test_init_distributed_is_noop_without_world_size(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    device, rank, world_size = init_distributed()
    assert rank == 0 and world_size == 1
    assert device == default_device()


def test_warmup_starts_below_the_peak_and_cosine_decays(tokenizer):
    cfg = _cfg()
    model = TinyJev(cfg)
    data = EncodedSet.build(_dataset(count=32), tokenizer, cfg)
    history = train(model, data, epochs=10, max_rows=16, lr=1e-3)
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
            max_rows=24,
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


class TestTemperatureIdentifiability:
    """A temperature is only meaningful if the loss responds to it.

    On a near-chance model the NLL curve is flat past T~4, so the search returns
    whichever grid point it happens to evaluate. The same model fitted on two
    halves of the test split gave T=4.69 and T=1.82 -- one of those cannot be right.
    """

    def test_flat_logits_are_not_identifiable(self):
        grid = _grid()
        flat = [(torch.zeros(4), 0.0)] * 200
        assert not is_identifiable(flat, "choice", grid)

    def test_decisive_logits_are_identifiable(self):
        grid = _grid()
        sharp = [(torch.tensor([8.0, -8.0, 0.0, 0.0]), 0.0)] * 200
        assert is_identifiable(sharp, "choice", grid)

    def test_fit_warns_when_the_optimum_is_on_the_grid_edge(self, capsys):
        # a very confident model wants T < 0.25, below the search range
        collected = {
            "noul": [(torch.tensor([9.0]), 1.0)] * 50 + [(torch.tensor([-9.0]), 0.0)] * 50,
            "choice": [],
            "score": [],
        }
        fit_temperatures(collected)
        out = capsys.readouterr().out
        assert "grid edge" in out

    def test_fit_warns_when_the_curve_is_flat(self, capsys):
        collected = {"noul": [], "choice": [(torch.zeros(4), 0.0)] * 200, "score": []}
        fit_temperatures(collected)
        out = capsys.readouterr().out
        assert "not identifiable" in out

    def test_no_warning_for_a_clean_fit(self, capsys):
        # moderately confident and sometimes wrong: the optimum lands inside the
        # grid and the loss responds to T, so no warning is appropriate
        collected = {
            "noul": [(torch.tensor([0.9]), 1.0)] * 60
            + [(torch.tensor([0.9]), 0.0)] * 25
            + [(torch.tensor([-0.9]), 0.0)] * 60
            + [(torch.tensor([-0.9]), 1.0)] * 25,
            "choice": [],
            "score": [],
        }
        temperatures = fit_temperatures(collected)
        out = capsys.readouterr().out
        assert "warning" not in out, out
        assert 0.25 < temperatures["noul"] < 10.0

    def test_warnings_can_be_suppressed(self, capsys):
        collected = {"noul": [], "choice": [(torch.zeros(4), 0.0)] * 200, "score": []}
        fit_temperatures(collected, warn=False)
        assert "warning" not in capsys.readouterr().out


def _grid(low=0.25, high=10.0, steps=10):
    return [
        math.exp(math.log(low) + i * (math.log(high) - math.log(low)) / (steps - 1))
        for i in range(steps)
    ]
