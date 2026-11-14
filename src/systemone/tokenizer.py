from typing import Protocol

PAD_ID = 0
SEP_ID = 1
_BYTE_BASE = 2


class Tokenizer(Protocol):
    vocab_size: int
    pad_id: int
    sep_id: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


def truncate_head_tail(ids: list[int], limit: int) -> list[int]:
    if len(ids) <= limit:
        return ids
    head = limit * 3 // 4
    return ids[:head] + ids[-(limit - head) :]


class ByteTokenizer:
    vocab_size = _BYTE_BASE + 256
    pad_id = PAD_ID
    sep_id = SEP_ID

    def encode(self, text: str) -> list[int]:
        return [b + _BYTE_BASE for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        return bytes(i - _BYTE_BASE for i in ids if i >= _BYTE_BASE).decode(
            "utf-8", errors="replace"
        )
