"""Browse the S1MB and bekko datasets in a local browser.

    uv run python scripts/browse.py

Everything is read from the local HuggingFace cache by file path -- the arrow
files and their dataset_info.json. No Hub call is made, so a config that was never
downloaded is reported as missing instead of being fetched.
"""

import argparse
import json
import os
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Belt and braces: even a stray `datasets` code path must not reach the network.
for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(name, "1")

from systemone.bpe import SentencePieceTokenizer
from systemone.paths import CHECKPOINT_DIR
from systemone.prompt import choice_query, noul_query, score_query
from systemone.s1mb import BEKKO_REPO, S1MB_REPO, parse_decision, render_state
from systemone.schema import Choice, Noul
from systemone.tokenizer import truncate_head_tail

HTML = Path(__file__).with_name("browse.html")
DEFAULT_TOKENIZER = CHECKPOINT_DIR / "bekko_tiny" / "tokenizer.model"
REPOS = {S1MB_REPO: "benchmark", BEKKO_REPO: "training"}
VOCAB_CASE_CAP = 400  # cases one vocabulary count reads


def plain(value) -> str:
    """Dataset text lives in JSON strings; unwrap it the way the loader does."""
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return "" if value is None else str(value)
    return parsed if isinstance(parsed, str) else json.dumps(parsed, ensure_ascii=False)


def cache_root(repo: str) -> Path:
    from datasets.config import HF_DATASETS_CACHE

    return Path(HF_DATASETS_CACHE) / repo.replace("/", "___")


def cached_configs(repo: str) -> list[str]:
    root = cache_root(repo)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def config_dir(repo: str, config: str) -> Path | None:
    """Newest cache directory for a config, or None if it was never downloaded."""
    candidates = sorted(
        (path.parent for path in (cache_root(repo) / config).glob("**/dataset_info.json")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def split_files(repo: str, config: str) -> dict[str, Path]:
    directory = config_dir(repo, config)
    if directory is None:
        return {}
    info = json.loads((directory / "dataset_info.json").read_text())
    files = {}
    for split in info.get("splits", {}):
        matches = sorted(directory.glob(f"*-{split}.arrow"))
        if matches:
            files[split] = matches[0]
    return files


def head_tail_split(length: int, limit: int) -> tuple[int, int]:
    """Mirror `truncate_head_tail`: 3/4 of the budget from the front, rest from the back."""
    if length <= limit:
        return length, 0
    head = limit * 3 // 4
    return head, limit - head


class Store:
    def __init__(self, tokenizer: SentencePieceTokenizer | None):
        self.tokenizer = tokenizer
        self._datasets: dict[tuple[str, str, str], object] = {}
        self._ids: dict[tuple[str, str, str], list[str]] = {}
        self._counts: dict[tuple[str, str, str], dict] = {}
        self._lock = threading.RLock()  # ids() re-enters dataset()

    def dataset(self, repo: str, config: str, split: str):
        key = (repo, config, split)
        with self._lock:
            if key not in self._datasets:
                from datasets import Dataset

                path = split_files(repo, config).get(split)
                if path is None:
                    raise FileNotFoundError(
                        f"{repo} · {config} · {split} is not in the local cache"
                    )
                self._datasets[key] = Dataset.from_file(str(path))
            return self._datasets[key]

    def ids(self, repo: str, config: str, split: str) -> list[str]:
        key = (repo, config, split)
        with self._lock:
            if key not in self._ids:
                self._ids[key] = [
                    str(case) for case in self.dataset(repo, config, split)["case_id"]
                ]
            return self._ids[key]

    def meta(self) -> dict:
        repos = []
        for repo, role in REPOS.items():
            configs = []
            for config in cached_configs(repo):
                splits = sorted(split_files(repo, config))
                if splits:
                    configs.append({"name": config, "splits": splits})
            repos.append({"name": repo, "role": role, "configs": configs})
        return {
            "repos": repos,
            "tokenizer": "none" if self.tokenizer is None else self.tokenizer.vocab_size,
        }

    def tokens(self, ids: list[int], budget: int) -> dict:
        head, tail = head_tail_split(len(ids), budget)
        decode = self.tokenizer.decode
        return {
            "total": len(ids),
            "budget": budget,
            "head": head,
            "tail": tail,
            "dropped": max(0, len(ids) - head - tail),
            "head_tokens": [decode([i]) for i in ids[:head]],
            "tail_tokens": [decode([i]) for i in ids[-tail:]] if tail else [],
            "ids": ids[:head] + (ids[-tail:] if tail else []),
        }

    def page(self, repo, config, split, offset, limit, query, jump) -> dict:
        ids = self.ids(repo, config, split)
        if jump:
            matches = [i for i, case in enumerate(ids) if case == jump]
            if not matches:
                matches = [i for i, case in enumerate(ids) if jump.lower() in case.lower()]
            if not matches:
                raise LookupError(f"no case matching {jump!r} in {config}")
            offset = max(0, matches[0] - limit // 2)
        index = list(range(len(ids)))
        if query:
            needle = query.lower()
            index = [i for i, case in enumerate(ids) if needle in case.lower()]
        window = index[offset : offset + limit]
        rows = self.dataset(repo, config, split)[window]
        items = []
        for position, source in enumerate(window):
            decisions = rows["input"][position]["decisions"]
            kinds = Counter(decision.get("type") or "unsupported" for decision in decisions)
            items.append(
                {
                    "case_id": ids[source],
                    "index": source,
                    "decisions": len(decisions),
                    "kinds": dict(kinds),
                    "state_chars": len(render_state(rows["input"][position]["state_json"])),
                }
            )
        return {
            "total": len(index),
            "all": len(ids),
            "offset": offset,
            "items": items,
            "jumped": bool(jump),
        }

    def case(self, repo, config, split, case_id, state_budget, query_budget) -> dict:
        ids = self.ids(repo, config, split)
        if case_id not in ids:
            raise LookupError(f"no case {case_id!r} in {config}")
        index = ids.index(case_id)
        row = self.dataset(repo, config, split)[index]
        state = render_state(row["input"]["state_json"])
        targets = {target["decision_id"]: target for target in row["targets"]}

        state_block = None
        if self.tokenizer is not None:
            state_block = self.tokens(self.tokenizer.encode(state), state_budget)

        decisions = [
            self.decision_block(config, state, decision, targets.get(decision["id"]), query_budget)
            for decision in row["input"]["decisions"]
        ]
        return {
            "repo": repo,
            "split": split,
            "config": config,
            "case_id": case_id,
            "index": index,
            "state": state,
            "state_tokens": state_block,
            "state_chars": len(state),
            "budget": {"state": state_budget, "query": query_budget},
            "fits": bool(state_block is None or state_block["dropped"] == 0)
            and all(item["fits"] for item in decisions),
            "decisions": decisions,
            "raw": {"input": row["input"], "targets": row["targets"]},
        }

    def decision_block(self, config, state, decision, target, query_budget: int) -> dict:
        import random

        block = {
            "id": decision.get("id"),
            "kind": decision.get("type") or "unsupported",
            "reason": None,
            "instructions": None,
            "options": [],
            "fits": True,
            "rows": [],
        }
        if target is None:
            block["reason"] = "no target row"
            return block
        example = parse_decision(
            state, decision, target, None, random.Random(0), benchmark=config, shuffle=False
        )
        if example is None:
            block["reason"] = "does not parse into a typed decision"
            return block

        question = example.question
        block["instructions"] = question.instructions
        block["label"] = example.label
        if self.tokenizer is not None:
            block["instruction_tokens"] = len(self.tokenizer.encode(question.instructions))

        if isinstance(question, Noul):
            rendered = {"text": None, "gold": example.label}
            if self.tokenizer is not None:
                rendered["query"] = self.tokens(
                    noul_query(self.tokenizer, question, query_budget), query_budget
                )
                block["fits"] = block["fits"] and rendered["query"]["dropped"] == 0
            block["distribution"] = list(example.distribution or [])
            block["rows"] = [rendered]
            return block

        gold_index = (
            question.criteria.index(example.label)
            if isinstance(example.label, str)
            else int(example.label)
        )
        for position, criterion in enumerate(question.criteria):
            option = {
                "index": position,
                "text": criterion,
                "gold": position == gold_index,
                "probability": example.distribution[position] if example.distribution else None,
                "value": example.values[position] if example.values else None,
            }
            if self.tokenizer is not None:
                render = (
                    choice_query(self.tokenizer, question, criterion, query_budget)
                    if isinstance(question, Choice)
                    else score_query(self.tokenizer, question, criterion, query_budget)
                )
                option["query"] = self.tokens(render, query_budget)
                option["fit"] = option["query"]["dropped"] == 0
                block["fits"] = block["fits"] and option["fit"]
            block["options"].append(option)
        block["label_index"] = gold_index
        return block

    def vocab(self, repo, config, split) -> dict:
        """Every piece in the vocabulary, with how often this config uses it.

        Counts come from the first VOCAB_CASE_CAP cases: state text plus
        instructions and option texts, which is where the tokenizer is exercised.
        Enough to spot pieces that never fire at all.
        """
        if self.tokenizer is None:
            raise LookupError("no tokenizer loaded, so there is no vocabulary to show")
        key = (repo, config, split)
        if key not in self._counts:
            dataset = self.dataset(repo, config, split)
            scanned = min(len(dataset), VOCAB_CASE_CAP)
            rows = dataset[0:scanned]
            counts: Counter = Counter()
            for row_input in rows["input"]:
                counts.update(self.tokenizer.encode(render_state(row_input["state_json"])))
                for decision in row_input["decisions"]:
                    counts.update(self.tokenizer.encode(plain(decision.get("instructions_json"))))
                    for criterion in decision.get("criteria") or []:
                        counts.update(
                            self.tokenizer.encode(plain(criterion.get("description_json")))
                        )
            self._counts[key] = {
                "counts": counts,
                "scanned": scanned,
                "cases": len(dataset),
                "total": sum(counts.values()),
            }
        record = self._counts[key]
        counts = record["counts"]
        sp = getattr(self.tokenizer, "sp", None)
        piece_of = sp.id_to_piece if sp is not None else lambda i: self.tokenizer.decode([i])
        items = [
            {
                "id": index,
                "piece": piece_of(index),
                "count": counts.get(index, 0),
                "text": self.tokenizer.decode([index]),
            }
            for index in range(self.tokenizer.vocab_size)
        ]
        used = sum(1 for item in items if item["count"])
        return {
            "size": self.tokenizer.vocab_size,
            "used": used,
            "unused": self.tokenizer.vocab_size - used,
            "source": getattr(self.tokenizer, "model_path", "tokenizer"),
            "items": items,
            "total_tokens": record["total"],
            "scanned": record["scanned"],
            "cases": record["cases"],
            "config": config,
            "split": split,
        }


class Handler(BaseHTTPRequestHandler):
    store: Store
    budgets = {"state": 192, "query": 64}

    def log_message(self, *args):
        pass

    def send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        one = lambda key, default=None: (query.get(key) or [default])[0]
        store = self.store
        try:
            if parsed.path == "/":
                body = HTML.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/meta":
                meta = store.meta()
                meta["budget"] = self.budgets
                self.send(meta)
                return
            if parsed.path == "/api/cases":
                self.send(
                    store.page(
                        one("repo"), one("config"), one("split"),
                        int(one("offset", 0)), min(200, int(one("limit", 50))),
                        one("q", "") or "", one("jump", "") or "",
                    )
                )
                return
            if parsed.path == "/api/case":
                self.send(
                    store.case(
                        one("repo"), one("config"), one("split"), one("case"),
                        int(one("state", self.budgets["state"])),
                        int(one("query", self.budgets["query"])),
                    )
                )
                return
            if parsed.path == "/api/vocab":
                self.send(store.vocab(one("repo"), one("config"), one("split")))
                return
            self.send({"error": "not found"}, 404)
        except (LookupError, FileNotFoundError) as error:
            self.send({"error": str(error)}, 404)
        except Exception as error:  # a broken case should not kill the browser
            self.send({"error": f"{type(error).__name__}: {error}"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state", type=int, default=192)
    parser.add_argument("--query", type=int, default=64)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    path = Path(args.tokenizer) if args.tokenizer else DEFAULT_TOKENIZER
    tokenizer = SentencePieceTokenizer(path) if path.exists() else None
    if tokenizer is None:
        print(f"no tokenizer at {path}; token view disabled")

    Handler.store = Store(tokenizer)
    Handler.budgets = {"state": args.state, "query": args.query}
    print(f"local cache only: { {repo: len(cached_configs(repo)) for repo in REPOS} }")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"browsing on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
