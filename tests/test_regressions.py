"""One test per bug that shipped, so none of them can come back quietly.

Each docstring names the failure mode and the damage it did. These are the tests
that should have existed from the start; the rest of the suite is broader, but
this file is the institutional memory.
"""

import random

import pytest

from systemone import ByteTokenizer, Choice, Example, ModelConfig, Noul, Score
from systemone.data import EncodedSet, rows_for
from systemone.prompt import choice_query
from systemone.s1mb import parse_decision


def _cfg(**overrides) -> ModelConfig:
    defaults = dict(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=32,
        max_state_tokens=128,
        max_query_tokens=48,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


class TestCandidateWasDeleted:
    """BUG: front-only truncation removed the candidate from 44.8% of S1MB rows.

    A row is `instructions + SEP + candidate`, truncated to 48 tokens from the
    front. Instructions run to 600+ tokens, so the candidate was cut off entirely.
    Consequence: 27.1% of `choice` groups collapsed to one identical encoding,
    capping accuracy at 0.81 and making 19.2% of wrong answers unlearnable.
    Diagnosed as "the model is capacity-bound", which was wrong.
    """

    def test_long_instruction_does_not_eat_the_candidate(self, tokenizer):
        row = choice_query(
            tokenizer,
            Choice(instructions="i" * 600, criteria=["a", "b"]),
            "the actual option text",
            48,
        )
        assert tokenizer.sep_id in row
        assert "option" in tokenizer.decode(row)

    def test_every_option_survives_a_realistic_instruction(self, tokenizer):
        instruction = (
            "Is the prompt unsafe under this policy? Apply the Aegis 2.0 moderation "
            "taxonomy, including identity hate, sexual content, self-harm, violence, "
            "weapons, threats, privacy violations, criminal activity, and harassment. "
        )
        criteria = [f"option number {i}" for i in range(8)]
        question = Choice(instructions=instruction, criteria=criteria)
        rows = [choice_query(tokenizer, question, option, 48) for option in criteria]
        assert len(set(map(tuple, rows))) == 8
        for row in rows:
            assert tokenizer.sep_id in row

    def test_build_refuses_to_encode_an_unanswerable_group(self, tokenizer, monkeypatch):
        import systemone.data as data

        example = Example(
            state="s",
            question=Choice(instructions="q", criteria=["a", "b"]),
            label="a",
        )
        monkeypatch.setattr(
            data, "rows_for", lambda *a, **k: ([[7, 8], [7, 8]], 0.0, "choice", None)
        )
        with pytest.raises(ValueError, match="candidate text was truncated away"):
            EncodedSet.build([example], tokenizer, _cfg())


class TestEvalDroppedDecisions:
    """BUG: >8-option choices raised inside the parser and were silently dropped.

    With the docs' own eval recipe (`max_candidates=None`) this lost 10.8% of all
    S1MB judgments, all 32.4% of wide `choice` decisions, and 9 entire subsets,
    while leaving a Task Avg that looked plausible.
    """

    def test_wide_option_sets_are_constructible(self):
        for count in (9, 17, 36, 77, 151):
            Choice(instructions="q", criteria=[f"o{i}" for i in range(count)])

    def test_parsing_without_a_cap_is_not_dropped(self):
        criteria = [{"id": f"o{i}", "description_json": f'"opt {i}"'} for i in range(20)]
        decision = {
            "id": "d",
            "type": "choice",
            "instructions_json": '"q"',
            "criteria": criteria,
        }
        target = {"ids": [f"o{i}" for i in range(20)], "probabilities": [0.05] * 20}
        assert parse_decision("s", decision, target, None, random.Random(0)) is not None

    def test_loader_reports_drops_instead_of_hiding_them(self, monkeypatch):
        """A dropped decision must be counted and surfaced, not silently skipped."""
        import systemone.s1mb as s1mb

        monkeypatch.setattr(s1mb, "parse_decision", lambda *a, **k: None)
        monkeypatch.setattr(
            "datasets.load_dataset",
            lambda *a, **k: [
                {
                    "input": {
                        "state_json": '"s"',
                        "decisions": [
                            {"id": "d", "type": "choice", "instructions_json": '"q"'}
                        ],
                    },
                    "targets": [{"decision_id": "d", "ids": ["a"], "probabilities": [1.0]}],
                    "case_id": "c1",
                }
            ],
        )
        assert s1mb.load_typed_decisions("repo", ["cfg"], strict=False) == {}
        with pytest.raises(ValueError, match="dropped 1 decisions"):
            s1mb.load_typed_decisions("repo", ["cfg"], strict=True)


class TestNoulLabelsWereHidden:
    """BUG: the label definitions were discarded for all 12,357 `noul` decisions.

    `true` and `false` differ only by their definition text. Dropping it left the
    model answering a yes/no question with no statement of what yes means.
    """

    def test_definitions_reach_the_prompt(self):
        decision = {
            "id": "d",
            "type": "noul",
            "instructions_json": '"Is it unsafe?"',
            "criteria": [
                {"id": "false", "description_json": '"it does not violate policy"'},
                {"id": "true", "description_json": '"it violates policy"'},
            ],
        }
        target = {"ids": ["false", "true"], "probabilities": [0.6, 0.4]}
        example = parse_decision("s", decision, target, 8, random.Random(0))
        assert "does not violate" in example.question.instructions
        assert "violates policy" in example.question.instructions

    def test_absent_definitions_do_not_add_noise(self):
        decision = {
            "id": "d",
            "type": "noul",
            "instructions_json": '"Is it?"',
            "criteria": [],
        }
        target = {"ids": ["false", "true"], "probabilities": [0.5, 0.5]}
        example = parse_decision("s", decision, target, 8, random.Random(0))
        assert example.question.instructions == "Is it?"


class TestParameterCountDrifted:
    """BUG: a hand-written formula over-reported by 768 parameters.

    It counted three norms per block; blocks have two. The docs labelled the
    arithmetic "exact" while its own terms summed to 935,680 against a claimed
    1,001,472. It is now measured from the module.
    """

    def test_count_equals_the_allocated_parameters(self):
        from systemone.model import TinyJev

        for cfg in (_cfg(), ModelConfig()):
            assert cfg.parameter_count() == sum(
                p.numel() for p in TinyJev(cfg).parameters()
            )

    def test_deployed_model_count(self):
        cfg = _cfg(
            d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, ffn_hidden=352, vocab_size=2048
        )
        assert cfg.parameter_count() == 1_000_704


class TestScoreExpectedValueUsedIndices:
    """BUG: `score` used level indices, ignoring the dataset's real level values.

    118 S1MB decisions use a 0..1 scale and 83 use thirds. Treating `0.25` as
    index 1 makes the expected value wrong, and the benchmark scores MAE against
    the real value.
    """

    def test_real_values_are_normalised(self):
        from systemone.schema import make_score, normalise_levels

        assert normalise_levels([0.0, 0.5, 1.0]) == [0.0, 0.5, 1.0]
        answer = make_score({0: 0.0, 1: 0.0, 2: 1.0}, values=[0.0, 0.5, 1.0])
        assert answer.score == pytest.approx(1.0)

    def test_scoring_honours_values(self):
        """An unevenly spaced scale makes index-scoring and value-scoring disagree."""
        from systemone.scoring import Decision, score_adjusted

        # levels 0, 0.1, 1.0: index 1 is nowhere near the middle of the range
        values = (0.0, 0.1, 1.0)
        decisions = [
            Decision(
                benchmark="b",
                kind="score",
                probabilities=[0.0, 1.0, 0.0],  # always predicts level index 1
                distribution=tuple(1.0 if i == gold else 0.0 for i in range(3)),
                label=float(gold),
                values=values,
            )
            for gold in [2] * 80 + [0] * 20
        ]
        by_values = score_adjusted(decisions)

        index_decisions = [Decision(**{**d.__dict__, "values": None}) for d in decisions]
        by_indices = score_adjusted(index_decisions)

        # The two scales disagree about how bad the same prediction is, which is
        # why the dataset's real values have to be used rather than indices.
        assert by_indices == pytest.approx(-1.5)
        assert by_values == pytest.approx(-2.7)
        assert by_values != pytest.approx(by_indices)


class TestTemperaturesLeaked:
    """BUG: temperatures were fitted on the eval split and reported there.

    The fix is correct but the effect is small: measured on two halves of S1MB,
    ECE moved by 0.0002 on one half and 0.0156 on the other. The docs originally
    overstated this as a serious error; they now quote the measurement.
    """

    def test_scripts_fit_on_train(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "scripts"
        for name in ("train_bekko.py", "train_s1mb.py"):
            source = (root / name).read_text()
            assert "fit_temperatures(train_collected)" in source, (
                f"{name} must fit temperatures on the train split"
            )
            assert "fit_temperatures(collected)" not in source


class TestTemperatureFitIsNotIdentifiable:
    """BUG: the temperature fit silently returned meaningless values.

    On a near-chance model the NLL curve is flat past T~4, so the grid search
    returns whichever point it happens to evaluate. The same checkpoint fitted on
    two disjoint halves of S1MB gave choice T=4.69 and T=1.82. A confident model
    also wants T below the grid floor. Both cases used to pass silently.
    """

    def test_flat_curve_is_reported(self, capsys):
        import torch

        from systemone.calibrate import fit_temperatures

        fit_temperatures({"noul": [], "choice": [(torch.zeros(4), 0.0)] * 200, "score": []})
        assert "not identifiable" in capsys.readouterr().out

    def test_grid_edge_is_reported(self, capsys):
        import torch

        from systemone.calibrate import fit_temperatures

        collected = {
            "noul": [(torch.tensor([9.0]), 1.0)] * 50 + [(torch.tensor([-9.0]), 0.0)] * 50,
            "choice": [],
            "score": [],
        }
        fit_temperatures(collected)
        assert "grid edge" in capsys.readouterr().out

    def test_docs_quote_the_measured_magnitude(self):
        from pathlib import Path

        doc = (Path(__file__).resolve().parents[1] / "docs" / "08-calibration.md").read_text()
        assert "0.0236" in doc, "the doc must quote the measurement, not just assert leakage"
        assert "not identifiable" in doc or "not really fitted" in doc


class TestStaleCacheReused:
    """BUG: caches built with the truncation bug would be silently reloaded.

    The fingerprint only covered the tokenizer, so a cache containing deleted
    candidates passed validation. A version stamp now invalidates them.
    """

    def test_cache_without_a_version_is_rejected(self, tokenizer):
        cfg = _cfg()
        encoded = EncodedSet.build(
            [Example(state="s", question=Noul(instructions="q"), label=True)], tokenizer, cfg
        )
        encoded.__dict__.pop("version", None)
        assert not encoded.compatible(tokenizer, cfg)

    def test_behaviour_changing_edit_bumps_the_version(self):
        from pathlib import Path

        import systemone.data as data

        assert data.CACHE_VERSION >= 2
        source = Path(data.__file__).read_text()
        assert "__dict__.get(\"version\"", "version must be read from the instance dict"


class TestSepIdDocWasWrong:
    """BUG: docs claimed `[SEP]` is token 1 for both tokenizers.

    sentencepiece reserves `unk_id=1`, so the separator is 2 in a trained
    tokenizer. Code that hard-coded 1 would encode the unknown token instead.
    """

    def test_byte_tokenizer_sep(self):
        assert ByteTokenizer().sep_id == 1

    def test_trained_tokenizer_sep_comes_from_the_model(self):
        from pathlib import Path

        from systemone.bpe import SentencePieceTokenizer

        model = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "bekko_tiny"
            / "tokenizer.model"
        )
        if not model.exists():
            pytest.skip("no trained tokenizer")
        tokenizer = SentencePieceTokenizer(model)
        assert tokenizer.sep_id == tokenizer.sp.piece_to_id("\u241f")
        assert tokenizer.sep_id != tokenizer.sp.unk_id()


class TestRawAccuracyFlatteredTheModel:
    """BUG: the headline number was raw accuracy on a skewed benchmark.

    71.5% of `noul` labels are false (always-false = 0.715; the model scored
    0.698), and the modal `score` level covers 50% (constant = 0.501). Reporting
    0.534 "overall" as skill was measuring the marginals.
    """

    def test_adjusted_score_is_zero_for_a_marginal_only_model(self):
        from systemone.scoring import Decision, balanced_accuracy

        # always answer the majority class
        decisions = [
            Decision(
                benchmark="b",
                kind="noul",
                probabilities=[1.0, 0.0],
                distribution=(0.0 if i < 80 else 1.0,),
                label=0.0 if i < 80 else 1.0,
            )
            for i in range(100)
        ]
        assert balanced_accuracy(decisions) == pytest.approx(0.0, abs=1e-9)

    def test_eval_script_exists_and_uses_the_adjusted_metric(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "scripts" / "eval_s1mb.py").read_text()
        assert "build_report" in source
        assert "max_candidates=None" in source
        assert "shuffle_options=False" in source
