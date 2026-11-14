import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = Path(
    os.environ.get("SYSTEMONE_CHECKPOINT_DIR", PROJECT_ROOT / "artifacts")
)


def checkpoint_path(name: str) -> Path:
    path = CHECKPOINT_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
