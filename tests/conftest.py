import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from systemone import ByteTokenizer, ModelConfig  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer() -> ByteTokenizer:
    return ByteTokenizer()


@pytest.fixture(scope="session")
def cfg() -> ModelConfig:
    """Small config: fast to build, same structure as the deployed model."""
    return ModelConfig(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=64,
        max_state_tokens=64,
        max_query_tokens=48,
    )


@pytest.fixture(scope="session")
def long_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=64,
        max_state_tokens=512,
        max_query_tokens=512,
    )


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0)


def pytest_collection_modifyitems(config, items):
    """Skip dataset-backed tests unless the cache is present.

    Slow tests are excluded by default via `addopts`; run them with `-m slow`.
    """
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    has_s1mb = (hub / "datasets--hotchpotch--s1mb-dataset").exists()
    if has_s1mb:
        return
    skip = pytest.mark.skip(reason="S1MB not in the HuggingFace cache")
    for item in items:
        if "slow" in item.keywords or "network" in item.keywords:
            item.add_marker(skip)
