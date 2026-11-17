"""Build and measure tokenizer versions for side-by-side comparison.

    uv run python scripts/build_tokenizers.py

The comparison models live under `artifacts/tokenizer_versions/<name>/` and are
picked up by `scripts/browse.py` through their `meta.json`. Reading the cached
bekko configs is the slow part (about half a minute at the 400-case cap); the
parsed corpus is pickled to `--cache` and reused, so a later variant is quick.
Existing models are measured rather than rebuilt unless `--force` is given.

Measurements are deliberately three-headed. A tokenizer can win on one and lose
on another, so each variant reports:

- `document_tokens`: a fixed `aegis2` prose document, the text the model reads;
- `whole_words`: common English words that encode to a single piece;
- `single_char_pieces`: slots spent on one character each.

`tokens_per_char` runs over a fixed sample of the repeated corpus, and
`tokens_per_char_distinct` over the deduplicated corpus: the first is comparable
with the historical numbers, the second is the language the model actually reads.
"""

import argparse
import gc
import json
import os
import pickle
import random
import tempfile
import time
from pathlib import Path
from typing import Any

# Belt and braces: even a stray `datasets` code path must not reach the network.
for _name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(_name, "1")

from systemone.bpe import (
    DEFAULT_VOCAB_SIZE,
    SentencePieceTokenizer,
    is_foreign,
    train_tokenizer,
)
from systemone.paths import CHECKPOINT_DIR
from systemone.corpus import (
    BEKKO_REPO,
    all_configs,
    collect_texts,
    parse_decision,
    render_state,
)

VERSIONS = CHECKPOINT_DIR / "tokenizer_versions"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "systemone_bekko_corpus.pkl"
SAMPLE_EVERY = 20  # one text in 20 goes into the tokens-per-character sample
# The comparison tokenizers v0..v3 were built with 400 cases per config, which
# is the 552,862-text corpus docs/04 measures. Keeping the cap makes the new
# variants comparable with them; it is not the training cap.
MAX_CASES = 400

# The corpus a variant trains on. `dedup` is what `collect_texts` now returns;
# `states` is the comparison that weights document prose hardest. The v0..v3
# models predate this script and are measured in place, so they carry no corpus.
VARIANTS: dict[str, dict[str, Any]] = {
    "v0_raw": {
        "description": "no filtering, repeated corpus",
        "corpus": None,
    },
    "v1_strip": {
        "description": "script strip only, repeated corpus",
        "corpus": None,
    },
    "v2_floor3": {
        "description": "strip + floor 3, repeated corpus",
        "corpus": None,
    },
    "v3_floor10": {
        "description": "strip + fold + floor 10, repeated corpus",
        "corpus": None,
    },
    "v4_dedup": {
        "description": "dedup + split digits + Greek cleanup (new default)",
        "corpus": "dedup",
        "strip_foreign_scripts": True,
        "min_character_count": 10,
        "split_digits": True,
    },
}
DEFAULT_ONLY = ["v4_dedup"]

# Rubric vocabulary is not what documents use. These are the words the model
# should not have to spell out when it reads prose. The same list the earlier
# measurements used, so the fractions stay comparable.
COMMON_WORDS = (
    "doctor", "legal", "cannabis", "popular", "become", "having", "friend",
    "either", "anything", "recreational", "naturally", "think", "about",
    "would", "know", "medical", "plant", "people", "years", "medicine",
    "because", "without",
)

# A fixed aegis2-style document, not a rubric string. Short enough to read as
# prose here and identical across variants, so the token count is a comparison
# rather than a description of whatever row happened to load first.
DOCUMENT = (
    "prompt: I don't think my doctor would know about them. response: I have "
    "a friend who is a doctor and he doesn't know anything about them either."
)


def cached_configs() -> list[str]:
    """Configs whose train split is actually on disk.

    `all_configs` also returns directories with no train split, which raise on
    load. Filtering by the train arrow is what the earlier measurements did and
    is why their corpus count is 552,862 rather than a slightly larger number.
    """
    from datasets.config import HF_DATASETS_CACHE

    root = Path(HF_DATASETS_CACHE) / BEKKO_REPO.replace("/", "___")
    have = {path.parts[-4] for path in root.glob("*/*/*/*-train.arrow")}
    if not have:
        raise SystemExit(f"no cached train splits under {root}")
    return [config for config in all_configs(BEKKO_REPO) if config in have]


def load_corpus(cache: Path) -> dict[str, Any]:
    """The deduplicated corpus plus a sample of the repeated one.

    Streaming matters. Materialising 552,862 parsed examples, or their 196M
    characters, gets this killed on a 16GB machine. Only what training and the
    measurements need is kept: the deduplicated corpus and one text in
    `SAMPLE_EVERY` to measure tokens per character.

    The texts come from the same rows and the same shared RNG state as
    `load_train`, so `dedup` is exactly what `collect_texts` now returns.
    That makes the comparison isolate the corpus mix and nothing else.
    """
    if cache.exists():
        with cache.open("rb") as handle:
            return pickle.load(handle)

    from datasets import load_dataset

    started = time.perf_counter()
    configs = cached_configs()
    rng = random.Random(0)
    distinct: dict[str, None] = {}
    sample: list[str] = []
    emitted = 0
    for position, config in enumerate(configs, 1):
        try:
            dataset = load_dataset(BEKKO_REPO, config, split="train")
        except Exception as error:
            print(f"skip config {config}: {type(error).__name__}", flush=True)
            continue
        for case, row in enumerate(dataset):
            if case >= MAX_CASES:
                break
            state = render_state(row["input"]["state_json"])
            targets = {target["decision_id"]: target for target in row["targets"]}
            examples = []
            for decision in row["input"]["decisions"]:
                target = targets.get(decision["id"])
                if target is None:
                    continue
                example = parse_decision(
                    state, decision, target, 8, rng, benchmark=config
                )
                if example is not None:
                    examples.append(example)
            for text in collect_texts({"row": examples}, deduplicate=False):
                distinct.setdefault(text, None)
                if emitted % SAMPLE_EVERY == 0:
                    sample.append(text)
                emitted += 1
        del dataset
        gc.collect()
        if position % 10 == 0 or position == len(configs):
            print(
                f"  {position}/{len(configs)} configs, {emitted:,} texts, "
                f"{len(distinct):,} distinct, {time.perf_counter() - started:.0f}s",
                flush=True,
            )

    payload = {
        "dedup": list(distinct),
        "sample": sample,
        "emitted": emitted,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump(payload, handle)
    print(
        f"corpus loaded in {time.perf_counter() - started:.0f}s "
        f"emitted={emitted:,} dedup={len(payload['dedup']):,} "
        f"sample={len(sample):,} -> {cache}",
        flush=True,
    )
    return payload


def measure(
    tokenizer: SentencePieceTokenizer,
    sample: list[str],
    distinct: list[str],
    document: str,
    *,
    build_seconds: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    bodies = [
        tokenizer.sp.id_to_piece(index).replace("\u2581", "")
        for index in range(tokenizer.vocab_size)
    ]
    singles = [body for body in bodies if len(body) == 1]
    chars = sum(len(text) for text in sample)
    tokens = sum(len(tokenizer.encode(text)) for text in sample)
    # The repeated corpus is the historical basis; the distinct corpus is the
    # language the model actually has to read, without the batch-layout repeats.
    distinct_chars = sum(len(text) for text in distinct)
    distinct_tokens = sum(len(tokenizer.encode(text)) for text in distinct)
    whole = [word for word in COMMON_WORDS if len(tokenizer.encode(word)) == 1]
    measured: dict[str, Any] = {
        "vocab_size": tokenizer.vocab_size,
        "single_char_pieces": len(singles),
        "non_ascii_singles": sum(
            1 for body in singles if any(ord(char) > 127 for char in body)
        ),
        "foreign_singles": sum(1 for body in singles if is_foreign(body)),
        "tokens_per_char": round(tokens / chars, 6),
        "tokens_per_char_sample": len(sample),
        "tokens_per_char_distinct": round(distinct_tokens / distinct_chars, 6),
        "document_chars": len(document),
        "document_tokens": len(tokenizer.encode(document)),
        "whole_words": len(whole),
        "whole_word_list": whole,
        "common_words_tested": len(COMMON_WORDS),
        "sep_id": tokenizer.sep_id,
        "pad_id": tokenizer.pad_id,
    }
    if build_seconds is not None:
        measured["seconds"] = round(build_seconds, 1)
    measured["measure_seconds"] = round(time.perf_counter() - started, 1)
    return measured


def build(
    name: str, spec: dict[str, Any], corpus: dict[str, Any], vocab_size: int
) -> tuple[SentencePieceTokenizer, float]:
    directory = VERSIONS / name
    directory.mkdir(parents=True, exist_ok=True)
    texts = corpus[spec["corpus"]]
    started = time.perf_counter()
    tokenizer = train_tokenizer(
        texts,
        directory / "tokenizer",
        vocab_size=vocab_size,
        strip_foreign_scripts=spec["strip_foreign_scripts"],
        min_character_count=spec["min_character_count"],
        split_digits=spec.get("split_digits", False),
    )
    seconds = time.perf_counter() - started
    print(
        f"{name}: trained {tokenizer.vocab_size} pieces from {len(texts):,} texts "
        f"in {seconds:.1f}s",
        flush=True,
    )
    return tokenizer, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=DEFAULT_ONLY,
        choices=sorted(VARIANTS),
        help=f"variants to build or measure (default: {' '.join(DEFAULT_ONLY)})",
    )
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--force", action="store_true", help="retrain even if tokenizer.model exists"
    )
    args = parser.parse_args()

    corpus = load_corpus(args.cache)
    summary_file = VERSIONS / "summary.json"
    summary = (
        json.loads(summary_file.read_text()) if summary_file.exists() else {}
    )
    summary = {name: meta for name, meta in summary.items() if name in VARIANTS}

    for name in args.only:
        spec = VARIANTS[name]
        model_file = VERSIONS / name / "tokenizer.model"
        seconds: float | None = None
        if model_file.exists() and not args.force:
            tokenizer = SentencePieceTokenizer(model_file)
            print(f"{name}: reusing {model_file}", flush=True)
        elif spec["corpus"] is None:
            raise SystemExit(f"{name}: no tokenizer at {model_file} to measure")
        else:
            tokenizer, seconds = build(name, spec, corpus, args.vocab_size)

        settings: dict[str, Any] = {"corpus": spec["corpus"]}
        if spec["corpus"] is not None:
            settings.update(
                strip_foreign_scripts=spec["strip_foreign_scripts"],
                min_character_count=spec["min_character_count"],
                split_digits=spec.get("split_digits", False),
                corpus_texts=len(corpus[spec["corpus"]]),
            )
        meta = {
            "label": name,
            "description": spec["description"],
            "settings": settings,
            **measure(
                tokenizer,
                corpus["sample"],
                corpus["dedup"],
                DOCUMENT,
                build_seconds=seconds,
            ),
        }
        (VERSIONS / name / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        summary[name] = meta
        print(
            f"{name}: doc={meta['document_tokens']} tokens/{meta['document_chars']} chars "
            f"whole={meta['whole_words']}/{meta['common_words_tested']} "
            f"singles={meta['single_char_pieces']} tpc={meta['tokens_per_char']}",
            flush=True,
        )

    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_file}", flush=True)


if __name__ == "__main__":
    main()
