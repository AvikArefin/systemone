"""Build the test-duplicate blocklist that `load_train` consumes.

Case ids and `input_hash` cannot find this overlap: across all 106 configs they
match the test split in zero rows, while the same case text really does appear
in train. A normalized content fingerprint finds it, and a per-config share
guard keeps shared templates (where many rows legitimately share one short
state) from being dropped along with the real duplicates.

Writes `artifacts/test_duplicates.json`, a `{config: [case_id]}` map.
"""

import json
import os
import sys
from pathlib import Path

from systemone.corpus import test_configs
from systemone.decontamination import find_test_duplicates, write_blocklist

OUT = Path(
    os.environ.get("SYSTEMONE_BLOCKLIST", "artifacts/test_duplicates.json")
)


def main() -> None:
    configs = sys.argv[1:] or test_configs()
    print(f"scanning {len(configs)} configs against the test split")
    duplicates = find_test_duplicates(configs)
    if not duplicates:
        print("no test duplicates found")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_blocklist(duplicates, OUT)
    total = sum(len(ids) for ids in duplicates.values())
    print(f"\nwrote {total} case ids across {len(duplicates)} configs to {OUT}")
    print(json.dumps({k: len(v) for k, v in sorted(duplicates.items())}, indent=2))


if __name__ == "__main__":
    main()
