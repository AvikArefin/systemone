import unicodedata
from collections import Counter
from pathlib import Path

import sentencepiece as spm

SEP_SYMBOL = "\u241f"

# A character must appear at least this often in the corpus to earn a vocabulary
# slot. Below it, `prune_rare` decides what to do -- fold, drop, or keep because
# its meaning cannot be recovered another way.
#
# 10 is measured. On the real corpus (552,862 texts, 196M characters, 1,565
# distinct) 175 characters fall below it: 80 fold to a base letter, 11 are
# invisible, and the rest are rare symbols. The high floor is deliberate: a
# vocabulary slot is expensive, and a pinyin tone mark or a stray `■` that appears
# five times in 196M characters will never earn it back.
# See docs/04-tokenizer-and-prompts.md for the sweep.
MIN_CHARACTER_COUNT = 10

# Unicode categories for characters that carry no visible mark: combining
# marks, format characters (zero-width joiner, bidi controls), and control
# characters. A rare one of these is noise with no rendering, so dropping it
# loses nothing -- unlike `∑`, which a reader would notice going missing.
INVISIBLE_CATEGORIES = frozenset({"Mn", "Mc", "Me", "Cf", "Cc", "Cs"})

# Currency signs. A decision task that mentions money must not render it as
# `<unk>`, so these are reserved whatever their frequency.
CURRENCY_SYMBOLS = "£¥€¢₹₱₽₩₪₫฿₴₦₡₲₵₸₺₼₾"

# Symbols that must keep a piece even when rare, because they cannot be spelled
# any other way. Reserved through `user_defined_symbols`, which SentencePiece
# guarantees regardless of frequency.
#
# Frequency alone is not enough: `∑` and `∩` both occur twice in the corpus, yet
# the final vocabulary cut kept `∩` and dropped `∑`, because BPE merges decide
# which candidates survive. A symbol has no alternative spelling, so `<unk>`
# there loses meaning rather than a typographic nicety.
#
# This is the maths vocabulary the decision tasks actually use: `aqua_rat`'s
# `√ ≤ π`, the operators in `helpsteer3`'s explanations, the set and arrow
# notation in retrieval rows. Only symbols present in the training corpus are
# reserved -- see `reserved_symbols`, because reserving an absent one would spend
# a slot on a piece the data never reaches.
MATHS_SYMBOLS = (
    "\u2211\u220f\u222b\u221a\u221e"          # ∑ ∏ ∫ √ ∞
    "\u2264\u2265\u2260\u2248\u00b1\u00d7\u00f7"  # ≤ ≥ ≠ ≈ ± × ÷
    "\u2208\u2209\u2229\u222a\u2282\u2283"    # ∈ ∉ ∩ ∪ ⊂ ⊃
    "\u2202\u2207\u2205"                      # ∂ ∇ ∅
    "\u2190\u2192\u2191\u2193\u2194\u21d2\u21d4"  # arrows
    "\u03c0\u03c3\u0394\u03b1\u03b2\u03bc\u03bb"  # π σ Δ α β μ λ
)

# Scripts this project never trains on. The corpora are English decision tasks,
# but a few configs carry stray foreign text -- a `Memo: cafe / 联系卡` disclaimer
# in open_jev__phone-extraction-control-v1, Chinese song titles in bge-hn-hotpotqa,
# Korean and Japanese article titles in dbpedia and fineweb. At
# `character_coverage=1.0` every one of those characters wins a permanent
# vocabulary slot, so a handful of rows spends ~300 of the 2048 slots on
# characters no English row will ever use.
FOREIGN_SCRIPTS = frozenset(
    {
        "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "HEBREW",
        "THAI", "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "KANNADA", "MALAYALAM",
        "GUJARATI", "GURMUKHI", "ORIYA", "SINHALA", "MYANMAR", "KHMER", "LAO",
        "TIBETAN", "ETHIOPIC", "CHEROKEE", "ARMENIAN", "GEORGIAN", "MONGOLIAN",
        "IDEOGRAPHIC", "HALFWIDTH",
    }
)

# Blocks that are pure emoji or decorative symbols. Kept as explicit ranges
# rather than a Unicode category test, because the categories do not split the
# way this needs: arrows and operators (`→`, `≤`, `∞`, `√`) are `Sm` and must
# survive -- `aqua_rat` and the relevance rows use them -- while `°`, `©`, `®`
# and `™` are `So` like the emoji but are ordinary English punctuation.
# Everything here is below 0x2600 or above 0x27BF, so all of those are safe.
EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),  # emoji, pictographs, emoticons, regional indicators
    (0x2600, 0x27BF),    # miscellaneous symbols and dingbats (✓ ✅ ❤ ☐ ♀ ♂)
    (0x2B00, 0x2BFF),    # extra arrows, stars and squares (⭐ ⬛)
    (0xFE00, 0xFE0F),    # variation selectors that make a glyph emoji
    (0x200D, 0x200D),    # zero width joiner
    (0x20E3, 0x20E3),    # combining enclosing keycap
)


def is_foreign(ch: str) -> bool:
    """True for a character the corpus should not train on.

    That is a character from a foreign script, or an emoji. Latin stays,
    including accents and typography (`é`, `£`, `—`, curly quotes): those turn up
    in real English rows. So do the math symbols, which are load-bearing in
    `aqua_rat`.
    """
    if ord(ch) < 128:
        return False
    name = unicodedata.name(ch, "")
    if not name:
        return True
    if name.split()[0] in FOREIGN_SCRIPTS:
        return True
    code = ord(ch)
    return any(low <= code <= high for low, high in EMOJI_RANGES)


def strip_foreign(text: str) -> str:
    """Drop foreign-script characters, keeping everything else in place."""
    return "".join(ch for ch in text if not is_foreign(ch))


def fold_char(ch: str) -> str | None:
    """The ASCII letter a rare accented character stands for, or None.

    `ǒ` becomes `o` and `Ñ` becomes `N`, so a rare accented name stays readable
    instead of collapsing to `<unk>`.

    Only letters fold. A symbol must not: `≠` decomposes to `≠`, which would strip
    to `=` and invert the meaning, and `∪` would become nothing. A symbol that
    matters is instead reserved in `reserved_symbols`.
    """
    if not unicodedata.category(ch).startswith("L"):
        return None
    decomposed = unicodedata.normalize("NFKD", ch)
    base = "".join(c for c in decomposed if not unicodedata.combining(c))
    if len(base) == 1 and base.isascii() and base != ch:
        return base
    return None


def prune_rare(text: str, minimum: int = MIN_CHARACTER_COUNT) -> str:
    """Normalise accented letters, then police whatever is still rare.

    Folding is not a frequency decision. `é` and `e` are the same letter; whether
    one appears 68,841 times or twice has no bearing on whether it deserves its
    own vocabulary slot. Gating folding on the floor was the bug that left 298
    accented letters -- `ý`, `é`, `ü`, `ö` -- sitting in the vocabulary beside
    their ASCII bases, which is two slots and two embedding rows for one letter.

    So two passes, in this order:

    1. Every accented letter folds to its base. Unconditional. This runs first so
       that the fold cannot push a character back over the floor and escape.
    2. Whatever is *still* non-ASCII and below the floor is resolved:
       invisible characters and rare symbols are dropped, unless reserved
       (currency, maths), where `<unk>` would lose meaning.

    Counts run over the whole text, so a character is judged on its total use
    rather than per line. The separator is exempt: it is a `user_defined_symbol`
    and must survive into the corpus for the trainer to reserve it.
    """
    # Pass 0: compose decomposed sequences first.
    #
    # Emails and scraped text arrive as `e` + U+0301 rather than `é`. Folding
    # cannot see that: `e` is ASCII and U+0301 has no base letter. SentencePiece
    # then applies NFKC itself, recomposing the pair into `é` and putting the
    # accented letter back into the vocabulary. Composing here means pass 1 sees
    # a real `é` and folds it. NFC, not NFKC: NFKC would also expand `ﬁ` and `⅓`,
    # changing text this is not supposed to touch.
    text = unicodedata.normalize("NFC", text)

    # Pass 1: fold every accented letter, however common. A translation table
    # built once from the corpus alphabet rather than a `fold_char` call per
    # character -- that was 89s on a 196M-character corpus, this is 6s.
    foldable: dict[str, str] = {}
    for char in set(text):
        base = fold_char(char)
        if base is not None:
            foldable[char] = base
    if foldable:
        text = text.translate(str.maketrans(foldable))

    # Pass 2: police what is still non-ASCII and rare, measured after folding.
    counts: Counter = Counter(text)
    keep = reserved_symbols(text)
    replacements: dict[str, str] = {}
    for char, n in counts.items():
        if ord(char) < 128 or n >= minimum or char == SEP_SYMBOL or char in keep:
            continue
        if unicodedata.category(char) in INVISIBLE_CATEGORIES:
            replacements[char] = ""
        else:
            # Rare non-ASCII with no reserved meaning: `■`, `【`, `⅓`, stray Greek.
            replacements[char] = ""
    if not replacements:
        return text
    return "".join(replacements.get(char, char) for char in text)


def reserved_symbols(text: str) -> list[str]:
    """Symbols that keep a piece whatever their frequency.

    Three groups, all of which lose meaning when rendered `<unk>`:

    - the separator, which the prompt format depends on;
    - currency (see `CURRENCY_SYMBOLS`);
    - maths notation present in the corpus (see `MATHS_SYMBOLS`).

    Narrowed to symbols this corpus actually contains: reserving an absent one
    still costs a vocabulary slot the data will never reach, which is the same
    dead-row problem the corpus filter exists to fix.
    """
    present = [symbol for symbol in CURRENCY_SYMBOLS if symbol in text]
    present += [symbol for symbol in MATHS_SYMBOLS if symbol in text]
    return [SEP_SYMBOL] + present


def clean_corpus(
    texts: list[str],
    *,
    strip_foreign_scripts: bool = True,
    min_character_count: int = MIN_CHARACTER_COUNT,
) -> str:
    """The corpus text handed to the trainer, after both filters.

    Two passes, because they answer different questions. `strip_foreign` removes
    characters from scripts this project never trains on -- unambiguous, no
    threshold. `prune_rare` handles characters that do belong to a wanted script
    but appear too few times to be worth a slot: the pinyin tone marks `Ÿ` and
    `ǒ` fold to `Y` and `o`, most Greek and stray IPA are left alone.
    """
    body = "\n".join(text.replace("\n", " ") for text in texts)
    if strip_foreign_scripts:
        body = strip_foreign(body)
    if min_character_count > 1:
        body = prune_rare(body, min_character_count)
    return body


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.pad_id = self.sp.pad_id()
        self.sep_id = self.sp.piece_to_id(SEP_SYMBOL)
        if self.pad_id < 0 or self.sep_id < 0:
            raise ValueError("tokenizer must define <pad> and the separator symbol")
        self.folds_accents = self._detect_folding()

    def _detect_folding(self) -> bool:
        """Whether this model was trained on accent-folded text.

        `clean_corpus` folds `é` to `e`, so a model built that way has no accented
        single-character pieces. Encoding raw `café` against it would otherwise
        produce an `<unk>` the model never saw in training -- a train/inference
        mismatch. Detecting it from the vocabulary keeps older checkpoints, which
        do have `é` pieces, working unchanged.
        """
        for char in "éèêëáàâäíìîïóòôöúùûüñç":
            if self.sp.piece_to_id(char) != self.sp.unk_id():
                return False
        return True

    def encode(self, text: str) -> list[int]:
        if self.folds_accents:
            # Mirror `prune_rare`'s first passes: compose, then fold. Without this
            # the model would meet an `<unk>` at inference that never appeared in
            # training, because the corpus was folded and the request was not.
            text = unicodedata.normalize("NFC", text)
            text = "".join(fold_char(char) or char for char in text)
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)


def train_tokenizer(
    texts: list[str],
    model_prefix: str | Path,
    vocab_size: int = 2048,
    *,
    strip_foreign_scripts: bool = True,
    min_character_count: int = MIN_CHARACTER_COUNT,
) -> SentencePieceTokenizer:
    prefix = Path(model_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus = prefix.with_suffix(".corpus.txt")
    body = clean_corpus(
        texts,
        strip_foreign_scripts=strip_foreign_scripts,
        min_character_count=min_character_count,
    )
    corpus.write_text(body)
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
        user_defined_symbols=reserved_symbols(body),
    )
    corpus.unlink()
    return SentencePieceTokenizer(prefix.with_suffix(".model"))
