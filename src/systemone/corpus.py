"""Load the bekko corpus: config rosters, typed rows, and tokenizer text.

The dataset has three splits. `train` is the only one training may read; `test`
is the evaluation set, and `validation` holds its 100-case samples.
"""

import json
import random
from functools import lru_cache
from pathlib import Path

from .data import Example
from .schema import Choice, Noul, Score


def _unquote(value: str) -> str:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _text(value: str) -> str:
    parsed = _unquote(value)
    if isinstance(parsed, str):
        return parsed
    return json.dumps(parsed, ensure_ascii=False)


def render_state(state_json: str) -> str:
    obj = json.loads(state_json)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "\n".join(
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in obj
        )
    if len(obj) == 1:
        only = next(iter(obj.values()))
        if isinstance(only, str):
            return only
    parts = []
    for key, value in obj.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts.append(f"{key}: {text}")
    return "\n".join(parts)


def _options(decision: dict) -> list[tuple[str, str]]:
    return [
        (criterion["id"], f"{criterion['id']}: {_text(criterion['description_json'])}")
        for criterion in decision.get("criteria") or []
    ]


def _subsample(
    options: list[tuple[str, str]],
    gold_id: str,
    limit: int | None,
    rng: random.Random,
    shuffle: bool = True,
) -> list[tuple[str, str]]:
    if limit is None:
        chosen = list(options)
        if shuffle:
            rng.shuffle(chosen)
        return chosen
    if len(options) <= limit:
        chosen = list(options)
        if shuffle:
            rng.shuffle(chosen)
        return chosen
    others = [option for option in options if option[0] != gold_id]
    rng.shuffle(others)
    chosen = [option for option in options if option[0] == gold_id] + others[: limit - 1]
    if shuffle:
        rng.shuffle(chosen)
    return chosen


def _normalise(weights: list[float]) -> tuple[float, ...] | None:
    total = sum(weights)
    if total <= 0.0:
        return None
    return tuple(weight / total for weight in weights)


def _criterion_definitions(decision: dict) -> str:
    """Render the label definitions a noul question is judged against.

    The dataset supplies a definition for each of `true` and `false`. Dropping
    them leaves the model guessing what the label means, because the two differ
    only in the definition text, not in the instruction.
    """
    parts = []
    for criterion in decision.get("criteria") or []:
        label = criterion.get("id")
        description = _text(criterion.get("description_json") or "")
        if label and description:
            parts.append(f"{label}: {description}")
    return "\n".join(parts)


def _drop_reason(decision: dict) -> str:
    """Name why a decision will not parse, for the drop warning.

    `type` is null for `ranking` decisions, which this model has no head for.
    Reporting those as `None` hides a whole subset's worth of data.
    """
    kind = decision.get("type")
    if kind is None:
        return f"unsupported:{decision.get('kind') or 'unknown'}"
    if kind not in ("noul", "choice", "score"):
        return f"unsupported:{kind}"
    return kind


def parse_decision(
    state: str,
    decision: dict,
    target: dict,
    max_candidates: int | None,
    rng: random.Random,
    benchmark: str | None = None,
    shuffle: bool = True,
) -> Example | None:
    kind = decision["type"]
    instructions = _text(decision["instructions_json"])
    probabilities = target.get("probabilities") or []
    ids = target.get("ids") or []
    if not probabilities or not ids:
        return None
    gold_position = probabilities.index(max(probabilities))
    gold_id = ids[gold_position]

    try:
        if kind == "noul":
            if "true" not in ids:
                return None
            definitions = _criterion_definitions(decision)
            if definitions:
                instructions = f"{instructions}\n{definitions}"
            probability = probabilities[ids.index("true")]
            return Example(
                state=state,
                question=Noul(instructions=instructions),
                label=probability >= 0.5,
                distribution=(probability,),
                benchmark=benchmark,
            )

        options = _options(decision)
        if kind == "choice":
            if not options or gold_id not in {option_id for option_id, _ in options}:
                return None
            chosen = _subsample(options, gold_id, max_candidates, rng, shuffle)
            weight_by_id = dict(zip(ids, probabilities))
            criteria = [text for _, text in chosen]
            distribution = _normalise([weight_by_id.get(oid, 0.0) for oid, _ in chosen])
            label = dict(options)[gold_id]
            return Example(
                state=state,
                question=Choice(instructions=instructions, criteria=criteria),
                label=label,
                distribution=distribution,
                benchmark=benchmark,
                candidate_ids=tuple(oid for oid, _ in chosen),
            )

        if kind == "score":
            ordered = sorted(
                decision.get("criteria") or [],
                key=lambda criterion: criterion.get("value")
                if criterion.get("value") is not None
                else 0.0,
            )
            if not 2 <= len(ordered) or (
                max_candidates is not None and len(ordered) > max_candidates
            ):
                return None
            levels = [_text(criterion["description_json"]) for criterion in ordered]
            values = [
                float(criterion["value"])
                if criterion.get("value") is not None
                else float(index)
                for index, criterion in enumerate(ordered)
            ]
            weights = [0.0] * len(ordered)
            position = {criterion["id"]: i for i, criterion in enumerate(ordered)}
            for target_id, probability in zip(ids, probabilities):
                if target_id.lstrip("-").isdigit():
                    index = int(target_id)
                else:
                    index = position.get(target_id, -1)
                if 0 <= index < len(weights):
                    weights[index] += probability
            distribution = _normalise(weights)
            label = weights.index(max(weights))
            return Example(
                state=state,
                question=Score(instructions=instructions, criteria=levels),
                label=label,
                distribution=distribution,
                benchmark=benchmark,
                candidate_ids=tuple(criterion["id"] for criterion in ordered),
                values=tuple(values),
            )
    except (ValueError, KeyError, StopIteration):
        return None
    return None


BEKKO_REPO = "hotchpotch/bekko-system-one-dataset-v0"

# One repo holds both rosters. Its `test` split is a verbatim copy of the
# published benchmark -- all 106 shared configs match row for row -- so training
# and evaluation never depend on two releases that can drift apart.


def all_configs(repo: str = BEKKO_REPO) -> list[str]:
    """Config names for a dataset repo, falling back to the local cache.

    `get_dataset_config_names` needs the Hub. Offline it does not fail -- it
    returns a single bogus `"default"` -- which is worse than failing, because a
    caller then loads nothing and reports zero cases. The cache directories are
    the real answer there.
    """
    from datasets import get_dataset_config_names
    from datasets.config import HF_DATASETS_CACHE

    try:
        names = list(get_dataset_config_names(repo))
    except Exception:
        names = []
    if names and names != ["default"]:
        return names

    root = Path(HF_DATASETS_CACHE) / repo.replace("/", "___")
    if not root.is_dir():
        return names
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _local_splits(repo: str) -> dict[str, set[str]]:
    """`{config: {split}}` from the Arrow files already in the local cache."""
    from datasets.config import HF_DATASETS_CACHE

    root = Path(HF_DATASETS_CACHE) / repo.replace("/", "___")
    if not root.is_dir():
        return {}
    splits: dict[str, set[str]] = {}
    for config in root.iterdir():
        if not config.is_dir():
            continue
        splits[config.name] = {
            path.name.rsplit("-", 1)[-1].removesuffix(".arrow")
            for path in config.glob("*/*/*.arrow")
        }
    return splits


@lru_cache(maxsize=None)
def configs_with_split(repo: str = BEKKO_REPO, split: str = "test") -> list[str]:
    """Config names that carry `split`.

    The Hub dataset card is one read for every config and split. Probing with
    `get_dataset_config_names` gives names only, and trying `load_dataset` per
    miss costs a round trip each -- bekko has 76 configs without a `test` split.
    Offline, the cached Arrow filenames already say which split they hold.
    """
    from huggingface_hub import HfApi

    try:
        card = HfApi().dataset_info(repo, files_metadata=False).card_data
        configs = card.to_dict().get("configs") if card is not None else None
    except Exception:
        configs = None
    if configs:
        return sorted(
            entry["config_name"]
            for entry in configs
            if entry.get("config_name")
            and any(file.get("split") == split for file in entry.get("data_files") or [])
        )
    local = _local_splits(repo)
    if local:
        return sorted(name for name, splits in local.items() if split in splits)
    return all_configs(repo)


def test_configs(repo: str = BEKKO_REPO) -> list[str]:
    """The evaluation roster: every config that has a `test` split (106)."""
    return configs_with_split(repo, "test")


def train_configs(repo: str = BEKKO_REPO) -> list[str]:
    """The training roster: every config that has a `train` split (153)."""
    return configs_with_split(repo, "train")


def load_typed_decisions(
    repo: str,
    configs: list[str],
    split: str = "test",
    max_candidates: int | None = 8,
    seed: int = 0,
    max_cases: int | None = None,
    shuffle_options: bool = True,
    strict: bool = False,
    exclude: dict[str, set[str]] | None = None,
) -> dict[str, list[Example]]:
    from datasets import load_dataset

    rng = random.Random(seed)
    grouped: dict[str, list[Example]] = {}
    dropped: dict[str, int] = {}
    skipped: dict[str, int] = {}
    for config in configs:
        try:
            dataset = load_dataset(repo, config, split=split)
        except Exception as error:
            print(f"skip config {config}: {type(error).__name__}")
            continue
        blocked = (exclude or {}).get(config, set())
        for case, row in enumerate(dataset):
            if max_cases is not None and case >= max_cases:
                break
            if row["case_id"] in blocked:
                skipped[config] = skipped.get(config, 0) + 1
                continue
            decisions = row["input"]["decisions"]
            targets = {target["decision_id"]: target for target in row["targets"]}
            state = render_state(row["input"]["state_json"])
            for decision in decisions:
                target = targets.get(decision["id"])
                if target is None:
                    continue
                example = parse_decision(
                    state,
                    decision,
                    target,
                    max_candidates,
                    rng,
                    benchmark=config,
                    shuffle=shuffle_options,
                )
                if example is None:
                    reason = _drop_reason(decision)
                    dropped[reason] = dropped.get(reason, 0) + 1
                    continue
                # namespace by config so case ids never merge across datasets
                grouped.setdefault(f"{config}/{row['case_id']}", []).append(example)
    if skipped:
        total = sum(skipped.values())
        print(f"excluded {total} benchmark-duplicate cases from training: {skipped}")
    if dropped:
        total = sum(dropped.values())
        message = f"dropped {total} decisions while parsing: {dropped}"
        if strict:
            raise ValueError(message)
        print(f"warning: {message}")
    return grouped


def load_test(
    configs: list[str] | None = None,
    max_candidates: int | None = 8,
    seed: int = 0,
    shuffle_options: bool = True,
    strict: bool = False,
) -> dict[str, list[Example]]:
    """The `test` split.

    `configs=None` uses `test_configs()`, not every config in the repo: the
    other 76 have no test split and would only produce skip warnings.
    """
    return load_typed_decisions(
        BEKKO_REPO,
        configs if configs is not None else test_configs(),
        split="test",
        max_candidates=max_candidates,
        seed=seed,
        shuffle_options=shuffle_options,
        strict=strict,
    )


def load_train(
    configs: list[str] | None = None,
    max_candidates: int | None = 8,
    seed: int = 0,
    max_cases: int | None = None,
    exclude: dict[str, set[str]] | None = None,
) -> dict[str, list[Example]]:
    """Training data, minus any test-duplicate cases in `exclude`.

    `exclude` comes from `find_test_duplicates` / `load_blocklist` in
    `decontamination`. Case ids and `input_hash` cannot stand in for it: across
    all 106 test configs they overlap in zero rows, while the same case text does
    appear in train. See `case_fingerprint`.
    """
    if configs is None:
        configs = train_configs()
    return load_typed_decisions(
        BEKKO_REPO,
        configs,
        split="train",
        max_candidates=max_candidates,
        seed=seed,
        max_cases=max_cases,
        exclude=exclude,
    )


def collect_texts(
    grouped: dict[str, list[Example]], *, deduplicate: bool = True
) -> list[str]:
    """The text a tokenizer should learn the language from.

    One string per typed example is not that text. A choice case with eight
    options becomes eight examples sharing one state and one instruction, so the
    same document is fed to BPE eight times and `collect_texts` emitted 552,862
    texts of which only 71,917 were distinct. BPE picks merges by frequency, so
    the merge budget goes to whatever repeats hardest: rubric descriptions
    repeat thousands of times and earn whole pieces (`▁instructions`,
    `▁probabilities`), while document prose repeats rarely and stays split
    (`do|ct|or`, `c|ann|ab|is`).

    Deduplicating keeps the *variety* of text the model reads -- every distinct
    state, instruction and criterion -- and removes repetition that comes from
    the batch layout rather than from the language. It costs one line and
    changes what the vocabulary is optimised for by design, not by accident.
    Measured on 148 bekko configs, an `aegis2` document encodes 13% shorter.
    See docs/04-tokenizer-and-prompts.md.

    `deduplicate=False` exists only to rebuild the historical comparison
    tokenizers (`v0_raw`..`v3_floor10`) that were trained on the repeated
    corpus; nothing in training should use it.
    """
    texts: list[str] = []
    for case in grouped.values():
        for example in case:
            texts.append(example.state)
            texts.append(example.question.instructions)
            if isinstance(example.question, (Choice, Score)):
                texts.extend(str(criterion) for criterion in example.question.criteria)
    if not deduplicate:
        return texts
    return list(dict.fromkeys(texts))
