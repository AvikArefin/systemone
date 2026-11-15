"""Tests for saving and reloading a checkpoint.

A checkpoint that loads with the wrong config, or silently drops its
temperatures, produces wrong answers with no error. These pin the round trip.
"""

import json


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
