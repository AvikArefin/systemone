"""Download the corpus, train the tokenizer, and build `.pt` caches — no training.

Safe to run on a CPU-only machine (e.g. a Kaggle CPU notebook) to prepare a
reusable cache directory before attaching it to a GPU training run.

Reads the same environment variables as `scripts/train_main.py`:
  SYSTEMONE_CHECKPOINT_DIR  base artifacts directory (default: <repo>/artifacts)
  SYSTEMONE_NAME            checkpoint/cache subfolder name (default: bekko_tiny)
  SYSTEMONE_STATE_TOKENS    state token budget (default: 192)
  SYSTEMONE_QUERY_TOKENS    query token budget (default: 64)
  BEKKO_CASES               max cases per config (default: 1000)
  BEKKO_CONFIG_LIMIT        optional cap on number of configs (default: 0 = all)
  SYSTEMONE_FIT_ONLY        set to 1 to drop examples that overflow the budget
  SYSTEMONE_INCLUDE_UNFIT   set to 1 to include configs in UNFIT_CONFIGS
"""

import os
import sys
from pathlib import Path

os.environ["SYSTEMONE_CACHE_ONLY"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_main  # noqa: E402


if __name__ == "__main__":
    train_main.main()
