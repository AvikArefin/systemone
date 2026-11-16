"""Browse the S1MB and bekko datasets in a local browser.

    uv run python scripts/browse.py              # s1mb test split
    uv run python scripts/browse.py --repo bekko # training data

Configs are opened lazily from the local HuggingFace cache; one config is read at
a time, and a case is parsed only when you click it. Nothing is downloaded.
"""

import argparse
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from systemone.bpe import SentencePieceTokenizer
from systemone.paths import CHECKPOINT_DIR
from systemone.prompt import choice_query, noul_query, score_query
from systemone.s1mb import BEKKO_REPO, S1MB_REPO, all_configs, parse_decision, render_state
from systemone.schema import Choice, Noul, Score
from systemone.tokenizer import truncate_head_tail

HTML = Path(__file__).with_name("browse.html")

DEFAULT_TOKENIZER = CHECKPOINT_DIR / "bekko_tiny" / "tokenizer.model"


def head_tail_split(length: int, limit: int) -> tuple[int, int]:
    """Mirror `truncate_head_tail`: 3/4 of the budget from the front, rest from the back."""
    if length <= limit:
        return length, 0
    head = limit * 3 // 4
    return head, limit - head


class Store:
    def __init__(self, repo: str, split: str, tokenizer: SentencePieceTokenizer | None):
        self.repo = repo
        self.split = split
        self.tokenizer = tokenizer
        self._datasets: dict[str, object] = {}
        self._ids: dict[str, list[str]] = {}
        self._lock = threading.RLock()  # ids() calls dataset() while holding it

    def dataset(self, config: str):
        with self._lock:
            if config not in self._datasets:
                from datasets import load_dataset

                self._datasets[config] = load_dataset(
                    self.repo, config, split=self.split
                )
            return self._datasets[config]

    def ids(self, config: str) -> list[str]:
        with self._lock:
            if config not in self._ids:
                self._ids[config] = [str(case) for case in self.dataset(config)["case_id"]]
            return self._ids[config]

    def tokens(self, ids: list[int], budget: int) -> dict:
        head, tail = head_tail_split(len(ids), budget)
        kept = ids[:head] + (ids[-tail:] if tail else [])
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

    def page(self, config: str, offset: int, limit: int, query: str) -> dict:
        ids = self.ids(config)
        index = list(range(len(ids)))
        if query:
            needle = query.lower()
            index = [i for i, case in enumerate(ids) if needle in case.lower()]
        window = index[offset : offset + limit]
        rows = self.dataset(config)[window]
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
        return {"total": len(index), "all": len(ids), "offset": offset, "items": items, "config": config}

    def case(self, config: str, case_id: str, state_budget: int, query_budget: int) -> dict:
        ids = self.ids(config)
        index = ids.index(case_id)
        row = self.dataset(config)[index]
        state = render_state(row["input"]["state_json"])
        targets = {target["decision_id"]: target for target in row["targets"]}

        state_block = None
        if self.tokenizer is not None:
            state_block = self.tokens(self.tokenizer.encode(state), state_budget)

        decisions = []
        for decision in row["input"]["decisions"]:
            target = targets.get(decision["id"])
            decisions.append(
                self.decision_block(config, state, decision, target, query_budget)
            )

        return {
            "config": config,
            "case_id": case_id,
            "index": index,
            "state": state,
            "state_tokens": state_block,
            "state_chars": len(state),
            "budget": {"state": state_budget, "query": query_budget},
            "fits": bool(
                state_block is None or state_block["dropped"] == 0
            ) and all(item["fits"] for item in decisions),
            "decisions": decisions,
            "raw": {"input": row["input"], "targets": row["targets"]},
        }

    def decision_block(self, config, state, decision, target, query_budget: int) -> dict:
        import random

        kind = decision.get("type")
        block = {
            "id": decision.get("id"),
            "kind": kind or "unsupported",
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
            state,
            decision,
            target,
            None,
            random.Random(0),
            benchmark=config,
            shuffle=False,
        )
        if example is None:
            block["reason"] = "does not parse into a typed decision"
            return block

        question = example.question
        block["instructions"] = question.instructions
        block["label"] = example.label if not isinstance(example.label, bool) else example.label

        if self.tokenizer is not None:
            block["instruction_tokens"] = len(self.tokenizer.encode(question.instructions))

        if isinstance(question, Noul):
            rows = [{"text": None, "gold": example.label}]
            if self.tokenizer is not None:
                ids = noul_query(self.tokenizer, question, query_budget)
                rows[0]["query"] = self.tokens(ids, query_budget)
                block["fits"] = block["fits"] and rows[0]["query"]["dropped"] == 0
            block["distribution"] = list(example.distribution or [])
            block["rows"] = rows
            return block

        gold_index = (
            question.criteria.index(example.label)
            if isinstance(question, Choice) or isinstance(example.label, str)
            else int(example.label)
        )
        for position, criterion in enumerate(question.criteria):
            option = {
                "index": position,
                "text": criterion,
                "gold": position == gold_index,
                "probability": (
                    example.distribution[position] if example.distribution else None
                ),
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


class Handler(BaseHTTPRequestHandler):
    store: Store
    budgets = {"state": 192, "query": 64}

    def log_message(self, *args):  # quiet: one line per request is noise
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
                self.send(
                    {
                        "repo": store.repo,
                        "split": store.split,
                        "configs": all_configs(store.repo),
                        "budget": self.budgets,
                        "tokenizer": (
                            "none" if store.tokenizer is None else store.tokenizer.vocab_size
                        ),
                    }
                )
                return
            if parsed.path == "/api/cases":
                self.send(
                    store.page(
                        one("config"),
                        int(one("offset", 0)),
                        min(200, int(one("limit", 50))),
                        one("q", "") or "",
                    )
                )
                return
            if parsed.path == "/api/case":
                state_budget = int(one("state", self.budgets["state"]))
                query_budget = int(one("query", self.budgets["query"]))
                self.send(store.case(one("config"), one("case"), state_budget, query_budget))
                return
            self.send({"error": "not found"}, 404)
        except Exception as error:  # a broken case should not kill the browser
            self.send({"error": f"{type(error).__name__}: {error}"}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=S1MB_REPO, choices=[S1MB_REPO, BEKKO_REPO])
    parser.add_argument("--split", default=None)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state", type=int, default=192, help="state token budget")
    parser.add_argument("--query", type=int, default=64, help="query token budget")
    parser.add_argument("--tokenizer", default=None, help="path to tokenizer.model")
    args = parser.parse_args()

    path = Path(args.tokenizer) if args.tokenizer else DEFAULT_TOKENIZER
    tokenizer = SentencePieceTokenizer(path) if path.exists() else None
    if tokenizer is None:
        print(f"no tokenizer at {path}; token view disabled")

    split = args.split or ("train" if args.repo == BEKKO_REPO else "test")
    Handler.store = Store(args.repo, split, tokenizer)
    Handler.budgets = {"state": args.state, "query": args.query}
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"browsing {args.repo} [{split}] on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
