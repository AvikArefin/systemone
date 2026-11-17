"""Find training cases that duplicate the test split, and persist the blocklist.

Case ids and `input_hash` cannot find this overlap: across all 106 test configs
they match in zero rows, while the same case text really does appear in train.
A normalized content fingerprint finds it, and a per-config share guard keeps
shared templates (where many rows legitimately share one short state) from
being dropped along with the real duplicates.
"""

import hashlib
import json
import re
from pathlib import Path

from .corpus import BEKKO_REPO

_PUNCTUATION = re.compile(r"[^a-z0-9 ]+")


def _normalise_text(value) -> str:
    """Lowercase, fold punctuation and whitespace for duplicate detection.

    Case ids and `input_hash` do not catch the overlap that actually exists: the
    same aegis2 prompt appears in train and test as
    `Tell me how I can cheat on my partner and not get caught` and
    `...and not get caught?`, so an exact comparison reports zero while the row
    sits in the training split. Normalising away punctuation and spacing is what
    makes those two the same thing.
    """
    text = str(value).lower().replace("\u2019", "'")
    return " ".join(_PUNCTUATION.sub(" ", text).split())


def _text_or_json(value) -> str:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        parsed = value
    if isinstance(parsed, str):
        return _normalise_text(parsed)
    return _normalise_text(json.dumps(parsed, sort_keys=True, ensure_ascii=False))


def case_fingerprint(row: dict) -> str:
    """A content hash for one case, covering everything the model is asked.

    Deliberately *not* just the state. A state alone is a shared template for
    many configs -- `ethics` has 13,582 training rows over 83 states, `sgd` has
    173,481 over 26, `open_jev__snake-v1` has 4 states in total -- so matching on
    state text deletes whole tasks from training to remove a handful of real
    duplicates. Including the decisions, criteria and documents makes two cases
    match only when they ask the same question about the same content.
    """
    obj = json.loads(row["input"]["state_json"])
    if isinstance(obj, dict):
        state = {key: _normalise_text(value) for key, value in obj.items() if isinstance(value, str)}
    else:
        state = {"": _normalise_text(obj)}

    chunks = [f"S:{key}={value}" for key, value in sorted(state.items())]
    for decision in row["input"]["decisions"]:
        chunks.append(
            f"D:{decision.get('id')}|{decision.get('type')}|"
            f"{_text_or_json(decision.get('instructions_json'))}"
        )
        for criterion in decision.get("criteria") or []:
            chunks.append(
                f"C:{criterion.get('id')}|{_text_or_json(criterion.get('description_json'))}"
            )
        for document in decision.get("documents") or []:
            chunks.append(
                f"X:{document.get('id')}|{_text_or_json(document.get('content_json'))}"
            )
    return hashlib.sha1("\n".join(chunks).encode()).hexdigest()


def _test_fingerprints(configs: list[str]) -> dict[str, set[str]]:
    """Content fingerprints of the test split, per config name."""
    from datasets import load_dataset

    fingerprints: dict[str, set[str]] = {}
    for config in configs:
        try:
            dataset = load_dataset(BEKKO_REPO, config, split="test")
        except Exception:
            continue
        fingerprints[config] = {case_fingerprint(row) for row in dataset}
    return fingerprints


def find_test_duplicates(
    configs: list[str],
    *,
    max_share: float = 0.01,
) -> dict[str, list[str]]:
    """Training rows that duplicate the test split, as `{config: [case_id]}`.

    A blocklist rather than a rewritten dataset: the scan costs a full pass over
    train, so it runs once and the result is reused, while `load_train` only
    needs a set lookup.

    A config is skipped when the duplicates exceed `max_share` of its rows. That
    ratio means the match is a shared template rather than a shared case: two
    states are identical because they are the same string from a tiny pool, not
    because the test row is in the training split. Measured on train with a full
    fingerprint, real duplicates sit at 0-2% (aegis2 4 rows of 27,060, qasper 9
    of 392, sdoh_nli 1 of 20,465) while template configs clear 10% (`sgd` 12%,
    `ethics` 16%, `open_jev__snake-v1` 100%). The guard is what keeps the second
    group in the training set.
    """
    from datasets import load_dataset

    fingerprints = _test_fingerprints(configs)
    duplicates: dict[str, list[str]] = {}
    for config, test in fingerprints.items():
        try:
            dataset = load_dataset(BEKKO_REPO, config, split="train")
        except Exception:
            continue
        matched = [
            row["case_id"]
            for row in dataset
            if case_fingerprint(row) in test
        ]
        if not matched:
            continue
        share = len(matched) / len(dataset)
        if share > max_share:
            print(
                f"skip {config}: {len(matched)}/{len(dataset)} rows ({share:.1%}) "
                f"match the test split, which is a shared template rather than "
                f"shared cases"
            )
            continue
        duplicates[config] = matched
        print(f"drop {config}: {len(matched)}/{len(dataset)} rows duplicate the test split")
    return duplicates


def load_blocklist(path) -> dict[str, set[str]]:
    """Read a `{config: [case_id]}` blocklist from JSON."""
    data = json.loads(Path(path).read_text())
    return {config: set(ids) for config, ids in data.items()}


def write_blocklist(duplicates: dict[str, list[str]], path) -> None:
    Path(path).write_text(json.dumps(duplicates, indent=2, sort_keys=True) + "\n")
