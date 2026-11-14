from pathlib import Path

import sentencepiece as spm

SEP_SYMBOL = "\u241f"


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.pad_id = self.sp.pad_id()
        self.sep_id = self.sp.piece_to_id(SEP_SYMBOL)
        if self.pad_id < 0 or self.sep_id < 0:
            raise ValueError("tokenizer must define <pad> and the separator symbol")

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)


def train_tokenizer(
    texts: list[str],
    model_prefix: str | Path,
    vocab_size: int = 2048,
) -> SentencePieceTokenizer:
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus = prefix.with_suffix(".corpus.txt")
    corpus.write_text("\n".join(text.replace("\n", " ") for text in texts))
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        user_defined_symbols=[SEP_SYMBOL],
    )
    corpus.unlink()
    return SentencePieceTokenizer(prefix.with_suffix(".model"))
