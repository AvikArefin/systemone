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
import unicodedata
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Belt and braces: even a stray `datasets` code path must not reach the network.
for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(name, "1")

from systemone.bpe import SentencePieceTokenizer, is_foreign
from systemone.paths import CHECKPOINT_DIR
from systemone.prompt import choice_query, noul_query, score_query
from systemone.s1mb import BEKKO_REPO, S1MB_REPO, parse_decision, render_state
from systemone.schema import Choice, Noul
from systemone.tokenizer import truncate_head_tail

HTML = Path(__file__).with_name("browse.html")
DEFAULT_TOKENIZER = CHECKPOINT_DIR / "bekko_tiny" / "tokenizer.model"
REPOS = {S1MB_REPO: "benchmark", BEKKO_REPO: "training"}
VOCAB_CASE_CAP = 400  # cases one vocabulary count reads
FOREIGN_SCAN_CASES = 250  # cases per config in the foreign-character scan


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


def piece_script(piece: str) -> str:
    """Classify a vocabulary piece so the sheet can show what it costs.

    `foreign` marks characters from a script the training corpus should never
    contain -- the ones `train_tokenizer` strips. `latin` keeps accented letters
    and typography, which do appear in English rows.
    """
    body = piece.replace("\u2581", "")
    if not body:
        return "marker"
    if all(ord(char) < 128 for char in body):
        return "ascii"
    if all(not is_foreign(char) for char in body):
        return "latin"
    return "foreign"


def script_of(char: str) -> str:
    """Uppercase Unicode script name, so a scan can group by `CYRILLIC` etc."""
    name = unicodedata.name(char, "")
    return name.split()[0] if name else "UNKNOWN"


def foreign_texts(row_input: dict) -> list[str]:
    """Every piece of text a row contributes, in the order the tokenizer sees it."""
    texts = [render_state(row_input["state_json"])]
    for decision in row_input["decisions"]:
        texts.append(plain(decision.get("instructions_json")))
        for criterion in decision.get("criteria") or []:
            texts.append(plain(criterion.get("description_json")))
        for document in decision.get("documents") or []:
            texts.append(plain(document.get("content_json")))
    return texts


def discover_tokenizers(explicit: Path | None) -> list[dict]:
    """Every tokenizer the browser can offer, newest layout first.

    Looks at the default checkpoint, any other `artifacts/*/tokenizer.model`,
    and whatever `--tokenizer` was given, so the comparison models built by
    `train_tokenizer` show up without editing the script.
    """
    seen: dict[Path, dict] = {}
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if DEFAULT_TOKENIZER.exists():
        candidates.append(DEFAULT_TOKENIZER)
    candidates.extend(sorted(CHECKPOINT_DIR.glob("*/tokenizer.model")))
    candidates.extend(sorted(CHECKPOINT_DIR.glob("*/*/tokenizer.model")))

    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        try:
            tokenizer = SentencePieceTokenizer(resolved)
        except Exception:
            continue
        pieces = [tokenizer.sp.id_to_piece(i) for i in range(tokenizer.vocab_size)]
        foreign_singles = sum(
            1 for piece in pieces
            if len(piece.replace("\u2581", "")) == 1
            and is_foreign(piece.replace("\u2581", ""))
        )
        try:
            label = str(resolved.parent.relative_to(CHECKPOINT_DIR))
        except ValueError:
            label = str(resolved.parent)

        # A version directory may describe itself, so the picker can say what
        # the model actually is rather than just its path.
        meta_file = resolved.parent / "meta.json"
        description = ""
        if meta_file.exists():
            try:
                description = json.loads(meta_file.read_text()).get("description", "")
            except (OSError, json.JSONDecodeError):
                description = ""

        seen[resolved] = {
            "key": label,
            "label": label,
            "description": description,
            "path": str(resolved),
            "vocab": tokenizer.vocab_size,
            "foreign_singles": foreign_singles,
            "pieces": pieces,
        }
    return list(seen.values())


class Store:
    def __init__(
        self,
        tokenizers: list[dict] | object | None = None,
        current: str | None = None,
    ):
        # A single tokenizer instance is allowed for tests and one-off scripts;
        # the server passes the discovery list instead.
        if tokenizers is not None and not isinstance(tokenizers, list):
            self.tokenizers = {"default": {"key": "default", "label": "default",
                                           "path": str(getattr(tokenizers, "model_path", "")),
                                           "vocab": tokenizers.vocab_size,
                                           "foreign_singles": 0, "pieces": []}}
            self._tokenizer = tokenizers
            self.current = "default"
        else:
            self.tokenizers = {entry["key"]: entry for entry in (tokenizers or [])}
            self._tokenizer = None
            self.current = current if current in self.tokenizers else next(
                iter(self.tokenizers), None
            )
        self._datasets: dict[tuple[str, str, str], object] = {}
        self._ids: dict[tuple[str, str, str], list[str]] = {}
        self._counts: dict[tuple[str, str, str, str], dict] = {}
        self._lock = threading.RLock()  # ids() re-enters dataset()

    @property
    def tokenizer(self) -> SentencePieceTokenizer | None:
        """The selected tokenizer, loaded on first use.

        Rebuilt rather than cached across a switch: a `SentencePieceProcessor`
        holds its vocabulary, so keeping two alive would only leak memory for a
        page that shows one at a time.
        """
        if self.current is None:
            return None
        entry = self.tokenizers[self.current]
        # A tokenizer passed straight in (tests, one-off scripts) is used as-is.
        if not entry["path"]:
            return self._tokenizer
        if self._tokenizer is None or getattr(
            self._tokenizer, "model_path", None
        ) != entry["path"]:
            self._tokenizer = SentencePieceTokenizer(entry["path"])
        return self._tokenizer

    def select_tokenizer(self, key: str) -> dict:
        if key not in self.tokenizers:
            raise LookupError(f"no tokenizer named {key!r}")
        self.current = key
        if self.tokenizers[key]["path"]:
            self._tokenizer = None
        return self.meta()

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
            "tokenizer": "none" if self.current is None else self.tokenizers[self.current]["vocab"],
            "tokenizers": [
                {
                    "key": entry["key"],
                    "label": entry["label"],
                    "description": entry.get("description", ""),
                    "vocab": entry["vocab"],
                    "foreign_singles": entry["foreign_singles"],
                }
                for entry in self.tokenizers.values()
            ],
            "tokenizer_key": self.current,
        }

    def tokens(self, ids: list[int], budget: int, total: int | None = None) -> dict:
        """Describe a token view; `total` is the length before any truncation.

        Prompt rows arrive already cut to the budget (`_render` truncates each
        segment), so `len(ids)` no longer says how much text there was. Passing
        the original count keeps the missed-token number and `fits` honest.
        """
        head, tail = head_tail_split(len(ids), budget)
        full = len(ids) if total is None else total
        decode = self.tokenizer.decode
        return {
            "total": full,
            "budget": budget,
            "head": head,
            "tail": tail,
            "dropped": max(0, full - head - tail),
            "head_tokens": [decode([i]) for i in ids[:head]],
            "tail_tokens": [decode([i]) for i in ids[-tail:]] if tail else [],
            "ids": ids[:head] + (ids[-tail:] if tail else []),
        }

    def row_tokens(
        self, instructions: str, candidate: str | None, ids: list[int], budget: int
    ) -> dict:
        """Token view for a prompt row, with its pre-truncation length restored."""
        total = len(self.tokenizer.encode(instructions))
        if candidate is not None:
            total += 1 + len(self.tokenizer.encode(candidate))  # separator
        return self.tokens(ids, budget, total=total)

    def foreign_scan(self, repo: str) -> dict:
        """Which configs contain foreign characters, and which script.

        Answers "where is the Cyrillic coming from" without opening config after
        config by hand. Capped per config so it stays a quick scan rather than a
        full pass over every dataset in the cache.
        """
        key = ("scan", repo)
        if key not in self._counts:
            results = {}
            for config in cached_configs(repo):
                splits = split_files(repo, config)
                split = "train" if "train" in splits else next(iter(splits), None)
                if split is None:
                    continue
                try:
                    dataset = self.dataset(repo, config, split)
                except Exception:
                    continue
                scanned = min(len(dataset), FOREIGN_SCAN_CASES)
                rows = dataset[0:scanned]
                scripts: Counter = Counter()
                rows_hit = 0
                for row_input in rows["input"]:
                    hit = False
                    for text in foreign_texts(row_input):
                        for char in text:
                            if is_foreign(char):
                                scripts[script_of(char)] += 1
                                hit = True
                    rows_hit += 1 if hit else 0
                if scripts:
                    results[config] = {
                        "split": split,
                        "scanned": scanned,
                        "rows": rows_hit,
                        "chars": sum(scripts.values()),
                        "scripts": dict(scripts.most_common()),
                    }
            self._counts[key] = results
        return {"configs": self._counts[key], "cap": FOREIGN_SCAN_CASES}

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
            kinds = Counter(
                decision.get("type") or decision.get("kind") or "unknown"
                for decision in decisions
            )
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
            "kind": decision.get("type") or decision.get("kind") or "unknown",
            "reason": None,
            "instructions": None,
            "options": [],
            "fits": True,
            "rows": [],
        }
        if target is None:
            block["reason"] = "no target row"
            return block
        if decision.get("type") is None and (
            decision.get("kind") == "ranking" or decision.get("documents")
        ):
            return self.ranking_block(decision, target, block)
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
                rendered["query"] = self.row_tokens(
                    question.instructions,
                    None,
                    noul_query(self.tokenizer, question, query_budget),
                    query_budget,
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
                option["query"] = self.row_tokens(
                    question.instructions, criterion, render, query_budget
                )
                option["fit"] = option["query"]["dropped"] == 0
                block["fits"] = block["fits"] and option["fit"]
            block["options"].append(option)
        block["label_index"] = gold_index
        return block

    def ranking_block(self, decision, target, block: dict) -> dict:
        """Show a `ranking` decision from its documents, not from the parser.

        The training parser drops these because the model has no ranking head,
        but the browser is a data viewer: `arxiv` and `agnews` are one ranking
        decision per case, so calling them "unsupported" hid entire configs.
        """
        block["instructions"] = plain(decision.get("instructions_json"))
        block["scoring"] = decision.get("scoring")
        weights = dict(zip(target.get("ids") or [], target.get("probabilities") or []))
        for position, document in enumerate(decision.get("documents") or []):
            document_id = document.get("id")
            probability = weights.get(document_id)
            block["options"].append(
                {
                    "index": position,
                    "id": document_id,
                    "text": plain(document.get("content_json")),
                    "gold": bool(probability),
                    "probability": probability,
                }
            )
        if block["options"]:
            block["label_index"] = max(
                range(len(block["options"])),
                key=lambda index: block["options"][index]["probability"] or 0.0,
            )
        if self.tokenizer is not None:
            block["instruction_tokens"] = len(
                self.tokenizer.encode(block["instructions"])
            )
        return block

    def vocab(self, repo, config, split) -> dict:
        """Every piece in the vocabulary, with how often this config uses it.

        Counts come from the first VOCAB_CASE_CAP cases: state text plus
        instructions, options and ranking documents, which is where the
        tokenizer is exercised. Enough to spot pieces that never fire at all.
        """
        if self.tokenizer is None:
            raise LookupError("no tokenizer loaded, so there is no vocabulary to show")
        key = (repo, config, split, self.current)
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
                    for document in decision.get("documents") or []:
                        counts.update(
                            self.tokenizer.encode(plain(document.get("content_json")))
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
                "script": piece_script(piece_of(index)),
            }
            for index in range(self.tokenizer.vocab_size)
        ]
        used = sum(1 for item in items if item["count"])
        # A single character from a foreign script is a slot no English row can
        # spend. Counting them makes the waste visible instead of a suspicion.
        foreign_singles = sum(
            1 for item in items
            if len(item["piece"].replace("\u2581", "")) == 1 and item["script"] == "foreign"
        )
        return {
            "size": self.tokenizer.vocab_size,
            "used": used,
            "unused": self.tokenizer.vocab_size - used,
            "foreign_singles": foreign_singles,
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
            if parsed.path == "/api/foreign":
                self.send(store.foreign_scan(one("repo")))
                return
            if parsed.path == "/api/tokenizer":
                self.send(store.select_tokenizer(one("key")))
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
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="start on this tokenizer; every other model found under artifacts is "
        "still selectable from the page",
    )
    args = parser.parse_args()

    explicit = Path(args.tokenizer) if args.tokenizer else None
    if explicit is not None and not explicit.exists():
        print(f"no tokenizer at {explicit}")
        explicit = None
    entries = discover_tokenizers(explicit)
    if not entries:
        print(f"no tokenizer found under {CHECKPOINT_DIR}; token view disabled")
    else:
        print(f"tokenizers: {', '.join(entry['label'] for entry in entries)}")

    Handler.store = Store(entries)
    Handler.budgets = {"state": args.state, "query": args.query}
    store = Handler.store
    print(f"local cache only: { {repo: len(cached_configs(repo)) for repo in REPOS} }")
    if store.current is not None:
        entry = store.tokenizers[store.current]
        print(f"tokenizer: {entry['label']} ({entry['vocab']} pieces, "
              f"{entry['foreign_singles']} foreign singles)")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"browsing on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
