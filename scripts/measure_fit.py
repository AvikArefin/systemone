"""Report configs that cannot fit the 1024/320 budget.

    uv run python scripts/measure_fit.py

Reads the local Arrow cache, takes the first three cases of every config, and
compares the longest state and query against the caps. A config whose row is
longer than the cap cannot be trained without truncating most of its text, so
`corpus.train_configs()` omits it by default (`corpus.UNFIT_CONFIGS`).

The script also reports drift from that constant, so a corpus change or a new
tokenizer cannot leave the mark silently stale.
"""

import argparse
import json
import os
from pathlib import Path

# Belt and braces: even a stray `datasets` code path must not reach the network.
for _name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(_name, "1")

from datasets import Dataset
from datasets.config import HF_DATASETS_CACHE

from systemone.bpe import SentencePieceTokenizer
from systemone.corpus import BEKKO_REPO, UNFIT_CONFIGS, render_state
from systemone.paths import CHECKPOINT_DIR

DEFAULT_TOKENIZER = CHECKPOINT_DIR / "bekko_tiny" / "tokenizer.model"
STATE_BUDGET = 1024
QUERY_BUDGET = 320
CASES = 3


def plain(value) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return "" if value is None else str(value)
    return parsed if isinstance(parsed, str) else json.dumps(parsed, ensure_ascii=False)


def split_files(repo: str, config: str) -> dict[str, Path]:
    """Arrow files for a config, from its newest local `dataset_info.json`."""
    root = Path(HF_DATASETS_CACHE) / repo.replace("/", "___")
    infos = sorted(
        root.glob(f"{config}/**/dataset_info.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not infos:
        return {}
    directory = infos[0].parent
    info = json.loads(infos[0].read_text())
    files = {}
    for split in info.get("splits", {}):
        matches = sorted(directory.glob(f"*-{split}.arrow"))
        if not matches:  # sharded: `...-train-00000-of-00002.arrow`
            matches = sorted(directory.glob(f"*-{split}-*-of-*.arrow"))
        if matches:
            files[split] = matches[0]
    return files


def measure(tokenizer, row_inputs) -> tuple[int, int]:
    """Longest state and longest query (instruction + separator + candidate)."""
    state = query = 0
    for row_input in row_inputs:
        state = max(state, len(tokenizer.encode(render_state(row_input["state_json"]))))
        for decision in row_input.get("decisions") or []:
            instruction = len(tokenizer.encode(plain(decision.get("instructions_json"))))
            candidates = [
                len(tokenizer.encode(plain(item.get("description_json"))))
                for item in decision.get("criteria") or []
            ] + [
                len(tokenizer.encode(plain(item.get("content_json"))))
                for item in decision.get("documents") or []
            ]
            if candidates:
                query = max(query, instruction + 1 + max(candidates))
            else:
                query = max(query, instruction)  # noul: instructions only
    return state, query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--cases", type=int, default=CASES)
    parser.add_argument("--state", type=int, default=STATE_BUDGET)
    parser.add_argument("--query", type=int, default=QUERY_BUDGET)
    args = parser.parse_args()

    tokenizer = SentencePieceTokenizer(args.tokenizer)
    root = Path(HF_DATASETS_CACHE) / BEKKO_REPO.replace("/", "___")
    configs = sorted(path.name for path in root.iterdir() if path.is_dir())

    measured: dict[str, tuple[str, int, int]] = {}
    unreadable = []
    for config in configs:
        splits = split_files(BEKKO_REPO, config)
        split = next((name for name in ("train", "test", "validation") if name in splits), None)
        if split is None:
            unreadable.append(config)
            continue
        try:
            dataset = Dataset.from_file(str(splits[split]))
            rows = dataset[0 : min(len(dataset), args.cases)]
        except Exception as error:
            unreadable.append(f"{config} ({type(error).__name__})")
            continue
        measured[config] = (split, *measure(tokenizer, rows["input"]))

    over = {
        config: (split, state, query)
        for config, (split, state, query) in measured.items()
        if state > args.state or query > args.query
    }

    print(
        f"tokenizer={args.tokenizer} cases/config={args.cases} "
        f"budget={args.state}/{args.query}"
    )
    print(f"scanned={len(measured)} unreadable={len(unreadable)} "
          f"over_budget={len(over)}\n")
    print(f"{'config':<58}{'split':>10}{'state':>8}{'query':>8}")
    for config in sorted(over):
        split, state, query = over[config]
        print(f"{config:<58}{split:>10}{state:>8}{query:>8}")

    expected = set(UNFIT_CONFIGS)
    found = set(over)
    if found == expected:
        print(f"\nmatches corpus.UNFIT_CONFIGS ({len(expected)} configs)")
    else:
        print("\ndrift from corpus.UNFIT_CONFIGS")
        for config in sorted(found - expected):
            print(f"  not marked: {config}")
        for config in sorted(expected - found):
            print(f"  marked but fits: {config}")
    if unreadable:
        print(f"unreadable: {', '.join(unreadable)}")


if __name__ == "__main__":
    main()
