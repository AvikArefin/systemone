import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .fingerprint import HasVocab, fingerprint
from .model import TinyJev
from .paths import checkpoint_path
from .tokenizer import ByteTokenizer

# Bumped when the checkpoint layout or its guards change meaning, so an old file
# is rejected rather than read with assumptions it does not satisfy.
CHECKPOINT_VERSION = 2


class VocabMismatch(RuntimeError):
    """A checkpoint's saved weights and its tokenizer disagree."""


def _require_match(
    saved: dict[str, Any],
    tokenizer: HasVocab,
    where: str,
    *,
    legacy_ok: bool = False,
) -> None:
    """Refuse weights and vocabulary that were not built together.

    `load_state_dict` checks tensor shapes only. Two tokenizers with the same
    `vocab_size` give the same embedding shape and different meanings -- token
    1146 is `Z` in one vocabulary and `▁interpretations` in another -- so a swap
    loads cleanly and silently mislabels every token. Only the fingerprint
    catches it.

    `legacy_ok` covers checkpoints written before this guard existed. It is safe
    for a checkpoint loaded by name, because `save_checkpoint` copies the
    tokenizer into the checkpoint directory beside `model.pt`, so the two were
    written together. It is not safe for a caller passing in an arbitrary
    tokenizer, which is why the default is strict.
    """
    expected = saved.get("tokenizer_fingerprint")
    if expected is None:
        if not legacy_ok:
            raise VocabMismatch(
                f"{where} was saved without a tokenizer fingerprint, so its "
                "vocabulary cannot be verified against the weights."
            )
        return
    actual = fingerprint(tokenizer)
    if actual != expected:
        raise VocabMismatch(
            f"{where} was trained with tokenizer {expected}, but the tokenizer in "
            f"use is {actual}. The weights and vocabulary mean different things; "
            "retrain rather than reuse."
        )


def save_checkpoint(
    name: str,
    model: TinyJev,
    tokenizer: HasVocab,
    temperatures: dict[str, float] | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    directory = checkpoint_path(name)
    state = {
        "model": model.state_dict(),
        "config": model.cfg,
        "temperatures": temperatures or {},
        "seed": seed,
        "parameter_count": model.cfg.parameter_count(),
        "extra": extra or {},
        "tokenizer_fingerprint": fingerprint(tokenizer),
        "checkpoint_version": CHECKPOINT_VERSION,
    }
    torch.save(state, directory / "model.pt")

    model_path = getattr(tokenizer, "model_path", None)
    if model_path is not None:
        source = Path(model_path)
        target = directory / "tokenizer.model"
        if source.resolve() != target.resolve():
            shutil.copy(source, target)
        tokenizer_meta = {
            "type": "sentencepiece",
            "file": target.name,
            "vocab_size": tokenizer.vocab_size,
            "fingerprint": fingerprint(tokenizer),
        }
    else:
        tokenizer_meta = {
            "type": "byte",
            "vocab_size": tokenizer.vocab_size,
            "fingerprint": fingerprint(tokenizer),
        }
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer_meta, indent=2))
    return directory


def load_checkpoint(name: str) -> tuple[TinyJev, HasVocab, dict[str, float]]:
    from .bpe import SentencePieceTokenizer

    directory = checkpoint_path(name)
    state = torch.load(directory / "model.pt", weights_only=False)
    meta = json.loads((directory / "tokenizer.json").read_text())

    # Build the tokenizer first, so the weights can be checked against it before
    # they are loaded into a model.
    if meta["type"] == "sentencepiece":
        tokenizer = SentencePieceTokenizer(directory / meta["file"])
    else:
        tokenizer = ByteTokenizer()

    # A checkpoint written before this guard has no fingerprint, but its tokenizer
    # was copied in beside the weights by `save_checkpoint`, so the pair is still
    # trustworthy. Record the fingerprint for next time, but never fail the load
    # over it: the checkpoint may be on a read-only mount.
    if state.get("tokenizer_fingerprint") is None:
        state["tokenizer_fingerprint"] = fingerprint(tokenizer)
        try:
            torch.save(state, directory / "model.pt")
        except OSError:
            pass
    _require_match(state, tokenizer, f"checkpoint {name!r}", legacy_ok=True)
    if "fingerprint" in meta and meta["fingerprint"] != fingerprint(tokenizer):
        raise VocabMismatch(
            f"checkpoint {name!r}: tokenizer.json records fingerprint "
            f"{meta['fingerprint']} but the vocabulary file hashes to "
            f"{fingerprint(tokenizer)}. It was replaced after the weights were saved."
        )

    model = TinyJev(state["config"])
    model.load_state_dict(state["model"])
    return model, tokenizer, state["temperatures"]
