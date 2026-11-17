"""Shared tokenizer identity, used to reject mismatched checkpoints and caches.

A vocabulary is a table of numbers to meaning. Two tokenizers with the same
`vocab_size` produce tensor-compatible weights that are semantically unrelated:
`load_state_dict` checks shapes only, so swapping vocabularies under a trained
model succeeds silently and every token then means the wrong thing. A fingerprint
is the cheap way to notice.
"""

import hashlib
from pathlib import Path
from typing import Protocol


class HasVocab(Protocol):
    vocab_size: int
    pad_id: int

    def encode(self, text: str) -> list[int]: ...


def fingerprint(tokenizer: HasVocab) -> str:
    """A short stable id for a tokenizer's vocabulary and merge table.

    A SentencePiece model hashes its file bytes, which covers both the pieces and
    the merges. A byte tokenizer has no file, so it is identified by its vocab
    size and a probe encoding instead -- enough to tell vocabularies apart.
    """
    digest = hashlib.sha256()
    model_path = getattr(tokenizer, "model_path", None)
    if model_path and Path(model_path).exists():
        digest.update(Path(model_path).read_bytes())
    else:
        probe = "the quick brown fox jumps over the lazy dog 0123456789 \u241f"
        digest.update(str(tokenizer.vocab_size).encode())
        digest.update(repr(tokenizer.encode(probe * 8)).encode())
    return digest.hexdigest()[:16]
