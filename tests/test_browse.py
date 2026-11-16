import importlib.util
import json
from pathlib import Path

import pytest

from systemone.tokenizer import ByteTokenizer


def load_browse():
    path = Path(__file__).resolve().parents[1] / "scripts" / "browse.py"
    spec = importlib.util.spec_from_file_location("browse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    """Enough of a datasets.Dataset for the store: column read and index list."""

    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        if key == "case_id":
            return [row["case_id"] for row in self.rows]
        if isinstance(key, list):
            return {
                column: [self.rows[index][column] for index in key]
                for column in ("case_id", "input", "targets")
            }
        return self.rows[key]


def row(case_id: str, state: str, instructions: str) -> dict:
    return {
        "case_id": case_id,
        "input": {
            "state_json": json.dumps(state),
            "decisions": [
                {
                    "id": "d1",
                    "type": "noul",
                    "instructions_json": instructions,
                    "criteria": [
                        {"id": "true", "description_json": "yes"},
                        {"id": "false", "description_json": "no"},
                    ],
                }
            ],
        },
        "targets": [
            {"decision_id": "d1", "ids": ["true", "false"], "probabilities": [0.7, 0.3]}
        ],
    }


@pytest.fixture()
def store():
    module = load_browse()
    store = module.Store("repo", "test", ByteTokenizer())
    store._datasets["cfg"] = FakeDataset(
        [row("cfg/a", "short state", "is it true?"), row("cfg/b", "x" * 500, "is it true?")]
    )
    return store


def test_head_tail_split_matches_truncation():
    module = load_browse()
    assert module.head_tail_split(10, 40) == (10, 0)
    head, tail = module.head_tail_split(1000, 40)
    assert (head, tail) == (30, 10)
    assert module.head_tail_split(40, 40) == (40, 0)


def test_page_lists_cases_and_filters(store):
    page = store.page("cfg", 0, 10, "")
    assert page["total"] == 2
    assert [item["case_id"] for item in page["items"]] == ["cfg/a", "cfg/b"]
    assert page["items"][0]["kinds"] == {"noul": 1}

    filtered = store.page("cfg", 0, 10, "b")
    assert [item["case_id"] for item in filtered["items"]] == ["cfg/b"]


def test_case_reports_truncation(store):
    short = store.case("cfg", "cfg/a", 192, 64)
    assert short["fits"] is True
    assert short["state_tokens"]["dropped"] == 0

    long = store.case("cfg", "cfg/b", 16, 64)
    assert long["fits"] is False
    assert long["state_tokens"]["dropped"] > 0
    assert long["state_tokens"]["head"] + long["state_tokens"]["tail"] == 16


def test_case_carries_the_decision_content(store):
    case = store.case("cfg", "cfg/a", 192, 64)
    decision = case["decisions"][0]
    assert decision["kind"] == "noul"
    assert decision["label"] is True
    assert decision["distribution"] == [0.7]
    assert decision["rows"][0]["query"]["total"] > 0
