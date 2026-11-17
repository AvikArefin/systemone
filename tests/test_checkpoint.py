"""Tests for saving and reloading a checkpoint.

A checkpoint that loads with the wrong config, or silently drops its
temperatures, produces wrong answers with no error. These pin the round trip.
"""

import json
from pathlib import Path

from systemone import ByteTokenizer, ModelConfig
from systemone.checkpoint import load_checkpoint, save_checkpoint
from systemone.model import TinyJev


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


def test_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_CHECKPOINT_DIR", str(tmp_path))
    import importlib

    import systemone.paths

    importlib.reload(systemone.paths)
    try:
        cfg = _cfg()
        model = TinyJev(cfg)
        tokenizer = ByteTokenizer()
        directory = save_checkpoint(
            "roundtrip",
            model,
            tokenizer,
            temperatures={"noul": 1.5, "choice": 2.0, "score": 1.2},
            seed=3,
            extra={"note": "hello"},
        )
        assert (directory / "model.pt").exists()
        assert (directory / "tokenizer.json").exists()

        loaded, loaded_tokenizer, temperatures = load_checkpoint("roundtrip")
        assert loaded.cfg == cfg
        assert isinstance(loaded_tokenizer, ByteTokenizer)
        assert temperatures == {"noul": 1.5, "choice": 2.0, "score": 1.2}
    finally:
        monkeypatch.undo()
        importlib.reload(systemone.paths)


def test_checkpoint_state_dict_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_CHECKPOINT_DIR", str(tmp_path))
    import importlib

    import systemone.paths

    importlib.reload(systemone.paths)
    try:
        model = TinyJev(_cfg())
        save_checkpoint("weights", model, ByteTokenizer())
        loaded, _, _ = load_checkpoint("weights")
        original = model.state_dict()
        restored = loaded.state_dict()
        assert set(original) == set(restored)
        for key in original:
            assert (original[key] == restored[key]).all(), f"{key} differs"
    finally:
        monkeypatch.undo()
        importlib.reload(systemone.paths)


def test_checkpoint_without_temperatures_defaults_to_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_CHECKPOINT_DIR", str(tmp_path))
    import importlib

    import systemone.paths

    importlib.reload(systemone.paths)
    try:
        save_checkpoint("plain", TinyJev(_cfg()), ByteTokenizer())
        _, _, temperatures = load_checkpoint("plain")
        assert temperatures == {}
    finally:
        monkeypatch.undo()
        importlib.reload(systemone.paths)


def test_tokenizer_metadata_records_the_type(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_CHECKPOINT_DIR", str(tmp_path))
    import importlib

    import systemone.paths

    importlib.reload(systemone.paths)
    try:
        directory = save_checkpoint("meta", TinyJev(_cfg()), ByteTokenizer())
        meta = json.loads((directory / "tokenizer.json").read_text())
        assert meta["type"] == "byte"
        assert meta["vocab_size"] == ByteTokenizer.vocab_size
    finally:
        monkeypatch.undo()
        importlib.reload(systemone.paths)


def test_parameter_count_is_stored(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_CHECKPOINT_DIR", str(tmp_path))
    import importlib

    import torch

    import systemone.paths

    importlib.reload(systemone.paths)
    try:
        cfg = _cfg()
        directory = save_checkpoint("params", TinyJev(cfg), ByteTokenizer())
        state = torch.load(directory / "model.pt", weights_only=False)
        assert state["parameter_count"] == cfg.parameter_count()
    finally:
        monkeypatch.undo()
        importlib.reload(systemone.paths)


def test_checkpoint_path_creates_the_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("SYSTEMONE_CHECKPOINT_DIR", str(tmp_path))
    import importlib

    import systemone.paths

    importlib.reload(systemone.paths)
    try:
        target = systemone.paths.checkpoint_path("new-model")
        assert target.exists() and target.is_dir()
    finally:
        monkeypatch.undo()
        importlib.reload(systemone.paths)


class TestWeightsAndVocabularyMustAgree:
    """BUG: a same-sized vocabulary swaps in silently.

    `load_state_dict` checks tensor shapes, not meaning. Two tokenizers with
    vocab_size 2048 produce the same embedding shape, so loading weights against
    the wrong one succeeds and every token then refers to the wrong piece -- token
    1146 is `Z` in one vocabulary and `▁interpretations` in another. Training
    continues and reports plausible numbers.
    """

    def _use_tmp_checkpoints(self, tmp_path, monkeypatch):
        """Point SYSTEMONE_CHECKPOINT_DIR at tmp_path and reload the path module."""
        import importlib

        import systemone.paths

        monkeypatch.setenv("SYSTEMONE_CHECKPOINT_DIR", str(tmp_path))
        importlib.reload(systemone.paths)

    def _restore(self, monkeypatch):
        import importlib

        import systemone.paths

        monkeypatch.undo()
        importlib.reload(systemone.paths)

    def test_save_records_a_fingerprint(self, tmp_path, monkeypatch):
        import torch

        self._use_tmp_checkpoints(tmp_path, monkeypatch)
        try:
            directory = save_checkpoint("finger", TinyJev(_cfg()), ByteTokenizer())
            state = torch.load(directory / "model.pt", weights_only=False)
            assert state["tokenizer_fingerprint"]
            meta = json.loads((directory / "tokenizer.json").read_text())
            assert meta["fingerprint"] == state["tokenizer_fingerprint"]
        finally:
            self._restore(monkeypatch)

    def test_a_swapped_tokenizer_file_is_refused(self, tmp_path, monkeypatch):
        """Replacing the vocabulary under saved weights must raise, not load."""
        import shutil

        import pytest

        from systemone import VocabMismatch
        from systemone.bpe import SentencePieceTokenizer

        self._use_tmp_checkpoints(tmp_path, monkeypatch)
        try:
            source = (
                Path(__file__).resolve().parents[1]
                / "artifacts" / "tokenizer_versions" / "v3_floor10" / "tokenizer.model"
            )
            other = (
                Path(__file__).resolve().parents[1]
                / "artifacts" / "tokenizer_versions" / "v1_strip" / "tokenizer.model"
            )
            if not source.exists() or not other.exists():
                pytest.skip("comparison tokenizers not built")

            tokenizer = SentencePieceTokenizer(source)
            cfg = _cfg(vocab_size=tokenizer.vocab_size)
            directory = save_checkpoint("swap", TinyJev(cfg), tokenizer)

            # Same vocabulary size, different contents -- the dangerous case.
            shutil.copy(other, directory / "tokenizer.model")
            with pytest.raises(VocabMismatch, match="mean different things"):
                load_checkpoint("swap")
        finally:
            self._restore(monkeypatch)

    def test_resume_refuses_a_different_vocabulary(self):
        """The training state carries the fingerprint; a mismatch must not load."""
        import pytest

        from systemone import ByteTokenizer, EncodedSet, Example, Noul
        from systemone.train import train

        cfg = _cfg()
        tokenizer = ByteTokenizer()
        examples = [
            Example(state="a state", question=Noul(instructions="is it true?"), label=True)
        ]
        dataset = EncodedSet.build(examples, tokenizer, cfg)
        model = TinyJev(cfg)

        resume = {
            "epoch": 1, "losses": [1.0], "lrs": [0.1], "evals": [],
            "model": model.state_dict(),
            "optimiser": {},
            "scheduler": {},
            "tokenizer_fingerprint": "some-other-vocabulary",
        }
        with pytest.raises(ValueError, match="cannot resume"):
            train(model, dataset, epochs=2, max_rows=8, resume=resume)

    def test_resume_refuses_a_state_without_a_fingerprint(self):
        """Pre-guard training state cannot be verified, so it is refused."""
        import pytest

        from systemone import ByteTokenizer, EncodedSet, Example, Noul
        from systemone.train import train

        cfg = _cfg()
        tokenizer = ByteTokenizer()
        dataset = EncodedSet.build(
            [Example(state="s", question=Noul(instructions="q?"), label=True)],
            tokenizer, cfg,
        )
        model = TinyJev(cfg)
        resume = {
            "epoch": 1, "losses": [], "lrs": [], "evals": [],
            "model": model.state_dict(), "optimiser": {}, "scheduler": {},
        }
        with pytest.raises(ValueError, match="predates the tokenizer fingerprint"):
            train(model, dataset, epochs=2, max_rows=8, resume=resume)
