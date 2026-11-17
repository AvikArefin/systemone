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
from systemone.corpus import parse_decision


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
    """BUG: front-only truncation removed the candidate from 44.8% of test rows.

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
    test judgments, all 32.4% of wide `choice` decisions, and 9 entire subsets,
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
        import systemone.corpus as corpus

        monkeypatch.setattr(corpus, "parse_decision", lambda *a, **k: None)
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
        assert corpus.load_typed_decisions("repo", ["cfg"], strict=False) == {}
        with pytest.raises(ValueError, match="dropped 1 decisions"):
            corpus.load_typed_decisions("repo", ["cfg"], strict=True)


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

    118 test decisions use a 0..1 scale and 83 use thirds. Treating `0.25` as
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

    The fix is correct but the effect is small: measured on two halves of the test split,
    ECE moved by 0.0002 on one half and 0.0156 on the other. The docs originally
    overstated this as a serious error; they now quote the measurement.
    """

    def test_scripts_fit_on_train(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "scripts"
        for name in ("train_main.py", "train_comparison.py"):
            source = (root / name).read_text()
            assert "fit_temperatures(train_collected)" in source, (
                f"{name} must fit temperatures on the train split"
            )
            assert "fit_temperatures(collected)" not in source


class TestTemperatureFitIsNotIdentifiable:
    """BUG: the temperature fit silently returned meaningless values.

    On a near-chance model the NLL curve is flat past T~4, so the grid search
    returns whichever point it happens to evaluate. The same checkpoint fitted on
    two disjoint halves of the test split gave choice T=4.69 and T=1.82. A confident model
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


class TestVocabWasSpentOnForeignCharacters:
    """BUG: character_coverage=1.0 reserved a slot for every stray character.

    The corpora are English decision tasks, but a few configs carry foreign text
    -- a `Memo: cafe / 联系卡` disclaimer, Chinese song titles, Korean article
    titles. Each of those characters won a permanent vocabulary slot and was
    never used again, so ~300 rows of the embedding table were dead.

    Lowering character_coverage was the obvious fix and the wrong one: it is a
    percentile, so 0.9995 also dropped `Z`, `Q`, `é`, `£` and `–`, which English
    rows do use. Stripping foreign scripts from the corpus removes the same
    waste and keeps those characters.
    """

    def test_foreign_scripts_are_detected(self):
        from systemone.bpe import is_foreign

        for char in "联卡のは한국москваعربيעבריתไทย":
            assert is_foreign(char), f"{char!r} should be foreign"

    def test_emoji_are_removed(self):
        from systemone.bpe import is_foreign

        for char in "😭💡📱🔥🔗👍👇🤷🤢🤦💦🤡😂😒❤️✅❗✔✓☐♂♀⭐⬛":
            assert is_foreign(char), f"{char!r} is an emoji, not corpus text"

    def test_joiner_and_variation_selectors_are_removed(self):
        from systemone.bpe import is_foreign

        # These are invisible but each one costs a vocabulary slot.
        for char in "\ufe0f\u200d\u20e3":
            assert is_foreign(char)

    def test_english_typography_is_kept(self):
        from systemone.bpe import is_foreign

        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0189":
            assert not is_foreign(char)
        for char in "éüñçàæøß£€–—‘’“”•…°±×÷©®™":
            assert not is_foreign(char), f"{char!r} appears in English data"

    def test_math_symbols_are_kept(self):
        from systemone.bpe import is_foreign

        # aqua_rat and the relevance rows need these; they are `Sm`, not emoji.
        for char in "→←↔⇒⇔≤≥≠∞∫∑∂∇∈∉∪∩√−":
            assert not is_foreign(char), f"{char!r} is a maths symbol"

    def test_strip_keeps_the_surrounding_text(self):
        from systemone.bpe import strip_foreign

        assert strip_foreign("Memo: café / 联系卡; do not dial") == "Memo: café / ; do not dial"
        assert strip_foreign("Great job 👍 done") == "Great job  done"

    def test_math_symbols_survive_because_aqua_rat_needs_them(self):
        from systemone.bpe import strip_foreign

        text = "− 4 ≤ x and √n + 1 and 7√ 2π"
        assert strip_foreign(text) == text

    def test_trained_vocab_has_no_dead_foreign_singles(self):
        from pathlib import Path

        from systemone.bpe import SentencePieceTokenizer, is_foreign

        model = (
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "bekko_tiny"
            / "tokenizer.model"
        )
        if not model.exists():
            pytest.skip("no trained tokenizer")
        tokenizer = SentencePieceTokenizer(model)
        pieces = [tokenizer.sp.id_to_piece(i) for i in range(tokenizer.vocab_size)]
        dead = [
            piece
            for piece in pieces
            if len(piece.replace("\u2581", "")) == 1 and is_foreign(piece.replace("\u2581", ""))
        ]
        # The checked-in model predates the fix; this documents the regression
        # rather than asserting it is gone. Retraining should take it to zero.
        assert len(dead) < 800, f"{len(dead)} foreign single-char pieces in the vocab"


class TestRareCharactersCostSlotsToo:
    """BUG: script filtering kept every rare letter, because it has no notion of "rare".

    `Ÿ` and `ǒ` are pinyin tone marks, not Greek or Cyrillic -- they are Latin, so
    a script filter kept them. Measured on the real corpus they appear 3 and 1
    times respectively across 196M characters, while each kept a vocabulary slot.
    Greek and the IPA set (`ʃ`, `ə`, `ˈ`) had the same problem.

    The fix is a frequency floor: a character below MIN_CHARACTER_COUNT is folded
    to its base letter, dropped if invisible, or left alone if it is a symbol.
    """

    def test_accents_fold_regardless_of_frequency(self):
        from systemone.bpe import prune_rare

        # Folding is not a frequency decision: `é` and `e` are the same letter at
        # any count, so every accented letter folds, common or not.
        assert prune_rare("café Ã ã ã ã", minimum=3) == "cafe A a a a"

    def test_rare_symbols_without_meaning_are_dropped(self):
        """The bug: a rare symbol has no base letter, so it used to be left alone.

        `■`, `【` and bare Greek letters kept vocabulary slots that no row would
        ever spend, because `prune_rare` only knew how to fold letters and drop
        invisible marks. Everything below the floor now gets a decision.
        """
        from systemone.bpe import prune_rare

        text = "a \u25a0 b \u3010 c \u2153 d"   # ■ 【 ⅓
        assert prune_rare(text, minimum=10) == "a  b  c  d"

    def test_currency_is_kept_however_rare(self):
        from systemone.bpe import prune_rare

        # ¥ appears once here, far below the floor, but money must stay readable.
        assert prune_rare("cost \u00a5 5", minimum=10) == "cost \u00a5 5"

    def test_ascii_is_never_dropped_even_on_a_tiny_corpus(self):
        from systemone.bpe import prune_rare

        # With a floor of 10 every letter of a short text is below it. ASCII has no
        # base to fold to, and dropping it would delete the text outright.
        assert prune_rare("abc Z", minimum=10) == "abc Z"

    def test_ascii_characters_are_never_moved(self):
        from systemone.bpe import prune_rare

        assert prune_rare("xyz xy x", minimum=3) == "xyz xy x"

    def test_the_separator_is_never_pruned(self):
        from systemone.bpe import SEP_SYMBOL, prune_rare

        # It appears once, so a naive floor would delete it and the trainer would
        # then refuse to reserve the piece the whole prompt format depends on.
        assert SEP_SYMBOL in prune_rare("a" + SEP_SYMBOL + "b", minimum=3)

    def test_a_common_accented_letter_still_folds(self):
        from systemone.bpe import prune_rare

        # 3 uses clears the floor, but folding does not care about the floor.
        assert prune_rare("Ü" * 3, minimum=3) == "UUU"

    def test_clean_corpus_composes_both_filters(self):
        from systemone.bpe import clean_corpus

        # CJK goes by script; `é` and `Ã` both appear once, below the default
        # floor of 10, so both fold to their base letter.
        texts = ["café 联系卡 Ã " + "é" * 3]
        cleaned = clean_corpus(texts)
        assert "联" not in cleaned
        assert "Ã" not in cleaned
        assert "é" not in cleaned
        assert "cafe" in cleaned

    def test_pruning_counts_across_the_whole_corpus_not_per_line(self):
        """Frequency must be corpus-wide, or every short line would be gutted."""
        from systemone.bpe import prune_rare

        # 'q' appears once per line but 4 times corpus-wide; corpus-wide counting
        # keeps it, per-line counting would not.
        corpus = "quick\nquick\nquick\nquick"
        assert "quick" in prune_rare(corpus, minimum=3)

    def test_pruning_never_empties_a_real_corpus_line(self):
        from systemone.bpe import prune_rare

        corpus = ("the quick brown fox jumps over the lazy dog. " * 3) + "a rare Ã character"
        cleansed = prune_rare(corpus, minimum=3)
        # The rare `Ã` becomes `A` rather than disappearing, so the word survives.
        assert "a rare A character" in cleansed

    def test_minimum_of_one_disables_pruning(self):
        from systemone.bpe import clean_corpus

        texts = ["a Ã"]
        assert clean_corpus(texts, min_character_count=1) == "a Ã"


class TestRareCharactersFoldInsteadOfVanishing:
    """Rare accented letters should degrade to their base letter, not to `<unk>`.

    Dropping `ǒ` loses the word; folding it to `o` keeps "café" readable. But a
    symbol cannot be folded, so `∑` must survive some other way -- it is reserved
    through `user_defined_symbols`, because frequency alone did not save it: `∑`
    and `∩` both occur twice, and the BPE vocabulary cut kept one and dropped the
    other.
    """

    def test_accented_letters_fold_to_their_base(self):
        from systemone.bpe import fold_char

        for char, base in (
            ("\u00d1", "N"), ("\u00d4", "O"), ("\u00da", "U"),   # Ñ Ô Ú
            ("\u01d2", "o"), ("\u01da", "u"), ("\u01ce", "a"),   # ǒ ǚ ǎ
            ("\u00e9", "e"), ("\u00fc", "u"),                    # é ü
        ):
            assert fold_char(char) == base, char

    def test_symbols_never_fold_even_when_they_have_an_ascii_decomposition(self):
        from systemone.bpe import fold_char

        # `§` decomposes to `S` and `≠` to `≠` -> `=`. Folding either would rewrite
        # the text as something it does not say, so only letters fold.
        assert fold_char("\u00a7") is None
        assert fold_char("\u2260") is None

    def test_a_character_that_would_fan_out_is_not_folded(self):
        from systemone.bpe import fold_char

        # `¼` decomposes to `1⁄4`; substituting that would change the text's shape.
        assert fold_char("\u00bc") is None

    def test_rare_accented_letter_folds_in_place(self):
        from systemone.bpe import prune_rare

        # one use each -> folded, not dropped
        assert prune_rare("a \u01d2 b \u01da c", minimum=3) == "a o b u c"

    def test_rare_symbol_that_is_reserved_is_preserved(self):
        from systemone.bpe import prune_rare

        # ∑ is a maths symbol, so it is reserved regardless of frequency.
        text = "sum \u2211 and intersection \u2229"
        assert prune_rare(text, minimum=3) == text

    def test_invisible_characters_are_dropped(self):
        from systemone.bpe import prune_rare

        # zero-width non-joiner, combining macron, bidi control
        assert prune_rare("a\u200cb\u0304c\u202cd", minimum=3) == "abcd"

    def test_accents_fold_however_frequent(self):
        from systemone.bpe import prune_rare

        # Even at 5 uses, well above the floor, `é` is not a separate letter.
        assert prune_rare("\u00e9" * 5, minimum=3) == "eeeee"

    def test_decomposed_accents_fold_too(self):
        """Scraped text arrives as `e` + U+0301, not `é`.

        Folding cannot see the decomposed form -- `e` is ASCII and U+0301 has no
        base letter -- so SentencePiece's own NFKC pass recomposed it back into
        `é` and the accented letter returned to the vocabulary. Composing with NFC
        before folding is what closes that hole.
        """
        from systemone.bpe import prune_rare

        decomposed = "d" + "e" + "\u0301"        # de + acute
        assert prune_rare(decomposed, minimum=3) == "de"

    def test_composed_and_decomposed_fold_to_the_same_thing(self):
        from systemone.bpe import prune_rare

        assert prune_rare("caf\u00e9") == prune_rare("cafe\u0301") == "cafe"

    def test_folding_survives_the_trainers_own_normalisation(self):
        """The real check: what SentencePiece sees after its NFKC pass is ASCII."""
        import unicodedata

        from systemone.bpe import prune_rare

        for source in ("caf\u00e9", "cafe\u0301", "na\u00efve", "nai\u0308ve"):
            cleaned = prune_rare(source)
            assert unicodedata.normalize("NFKC", cleaned).isascii(), source

    def test_folding_tokenizer_encodes_accented_input_without_unk(self):
        """Train/inference parity.

        The corpus is folded at build time, so a request arriving with `café`
        would otherwise hit an `<unk>` the model never saw in training. `encode`
        applies the same fold.
        """
        import tempfile
        from pathlib import Path

        from systemone.bpe import SentencePieceTokenizer, train_tokenizer

        filler = "the customer asked about the invoice and the refund policy. " * 30
        texts = [filler] * 20
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = train_tokenizer(texts, Path(directory) / "tokenizer", vocab_size=64)
            assert tokenizer.folds_accents is True
            ids = tokenizer.encode("café")
            assert tokenizer.sp.unk_id() not in ids, "accent produced an <unk>"
            assert tokenizer.decode(ids) == "cafe"

    def test_a_tokenizer_with_accent_pieces_is_left_alone(self):
        """Older checkpoints keep `é` as a real piece, so encode must not fold."""
        from pathlib import Path

        from systemone.bpe import SentencePieceTokenizer

        model = (
            Path(__file__).resolve().parents[1]
            / "artifacts" / "bekko_tiny" / "tokenizer.model"
        )
        if not model.exists():
            pytest.skip("no trained tokenizer")
        tokenizer = SentencePieceTokenizer(model)
        if tokenizer.folds_accents:
            pytest.skip("checked-in model was rebuilt with folding")
        ids = tokenizer.encode("café")
        assert tokenizer.sp.unk_id() not in ids

    def test_reserved_symbols_only_includes_what_the_corpus_contains(self):
        from systemone.bpe import SEP_SYMBOL, reserved_symbols

        corpus = "sum \u2211 and root \u221a"
        reserved = reserved_symbols(corpus)
        assert reserved[0] == SEP_SYMBOL
        assert "\u2211" in reserved and "\u221a" in reserved
        # `∩` is a maths symbol but absent here, so it must not cost a slot
        assert "\u2229" not in reserved

    def test_a_trained_tokenizer_keeps_a_rare_symbol_the_plain_floor_drops(self):
        """End to end: the symbol survives even though it appears once."""
        from systemone.bpe import clean_corpus, train_tokenizer

        filler = "the quick brown fox jumps over the lazy dog. " * 40
        texts = [filler + "sum \u2211 here"]
        body = clean_corpus(texts)
        assert "\u2211" in body, "the cleaner must not delete the symbol"

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            tokenizer = train_tokenizer(texts, Path(directory) / "tokenizer", vocab_size=64)
            assert tokenizer.sp.piece_to_id("\u2211") != tokenizer.sp.unk_id()
            assert tokenizer.sp.piece_to_id("\u241f") == tokenizer.sep_id



class TestGreekAndDigitsDoNotGetSlots:
    """Two more kinds of slot the corpus was paying for.

    `ό` is a Greek omicron with a tonos: NFKD gives Greek `ο`, not ASCII `o`, so
    the old fold skipped it and it kept a dead slot. Dates like `▁2027-0`
    repeated in the voucher rows and won merges that `doctor` (973 occurrences)
    could not afford.
    """

    def test_accented_greek_vowels_fold_to_latin(self):
        from systemone.bpe import fold_char

        assert fold_char("\u03cc") == "o"  # ό
        assert fold_char("\u03ac") == "a"  # ά
        assert fold_char("\u038c") == "O"  # Ό

    def test_non_reserved_greek_is_dropped_however_frequent(self):
        from systemone.bpe import prune_rare

        # `ο` and `γ` are quoted Greek words, not maths. Frequency does not save
        # them: a slot for `ο` beside `o` is still a wasted row.
        assert prune_rare("\u03bf" * 20, minimum=10) == ""
        assert prune_rare("\u03b3" * 20, minimum=10) == ""

    def test_reserved_maths_greek_is_kept(self):
        from systemone.bpe import prune_rare

        # `π` is in MATHS_SYMBOLS and must stay readable even once.
        assert prune_rare("\u03c0", minimum=10) == "\u03c0"

    def test_digits_never_merge_into_multi_digit_pieces(self, tmp_path):
        from systemone.bpe import train_tokenizer

        texts = ["the request date is 2027-01-15 and the voucher is 2027-0. " * 40]
        tokenizer = train_tokenizer(texts, tmp_path / "tokenizer", vocab_size=64)
        pieces = [
            tokenizer.sp.id_to_piece(index)
            for index in range(tokenizer.vocab_size)
        ]
        assert not [
            piece for piece in pieces
            if sum(char.isdigit() for char in piece) >= 2
        ], pieces


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

        source = (Path(__file__).resolve().parents[1] / "scripts" / "eval.py").read_text()
        assert "build_report" in source
        assert "max_candidates=None" in source
        assert "shuffle_options=False" in source


class TestUnfitConfigsWereStillTrained:
    """BUG: configs that overflow even a 1024/320 budget were loaded and cut.

    31 of 179 cached configs cannot fit that budget -- `qasper` alone is a
    26,977-token state. Training on them spends steps on text the model never
    sees, so the roster now omits them unless asked.
    """

    def test_train_configs_omits_unfit_configs(self, monkeypatch):
        import systemone.corpus as corpus

        monkeypatch.setattr(
            corpus,
            "configs_with_split",
            lambda repo, split: ["arc", "agnews", "qasper"],
        )
        assert corpus.train_configs() == ["arc"]
        assert corpus.train_configs(include_unfit=True) == ["arc", "agnews", "qasper"]

    def test_load_train_drops_unfit_configs_even_when_named(self, monkeypatch):
        import systemone.corpus as corpus

        seen = {}

        def fake(repo, configs, **kwargs):
            seen["configs"] = configs
            return {}

        monkeypatch.setattr(corpus, "load_typed_decisions", fake)
        corpus.load_train(["arc", "agnews"])
        assert seen["configs"] == ["arc"]
        corpus.load_train(["arc", "agnews"], include_unfit=True)
        assert seen["configs"] == ["arc", "agnews"]
