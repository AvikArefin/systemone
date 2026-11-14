import json
import shutil
from pathlib import Path

import torch

from .model import TinyJev
from .paths import checkpoint_path
from .tokenizer import ByteTokenizer


def save_checkpoint(
    name: str,
    model: TinyJev,
    tokenizer,
    temperatures: dict[str, float] | None = None,
    seed: int | None = None,
    extra: dict | None = None,
) -> Path:
    directory = checkpoint_path(name)
    state = {
        "model": model.state_dict(),
        "config": model.cfg,
        "temperatures": temperatures or {},
        "seed": seed,
        "parameter_count": model.cfg.parameter_count(),
        "extra": extra or {},
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
        }
    else:
        tokenizer_meta = {"type": "byte", "vocab_size": tokenizer.vocab_size}
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer_meta, indent=2))
    return directory


def load_checkpoint(name: str) -> tuple[TinyJev, object, dict[str, float]]:
    from .bpe import SentencePieceTokenizer

    directory = checkpoint_path(name)
    state = torch.load(directory / "model.pt", weights_only=False)
    model = TinyJev(state["config"])
    model.load_state_dict(state["model"])
    meta = json.loads((directory / "tokenizer.json").read_text())
    if meta["type"] == "sentencepiece":
        tokenizer = SentencePieceTokenizer(directory / meta["file"])
    else:
        tokenizer = ByteTokenizer()
    return model, tokenizer, state["temperatures"]
