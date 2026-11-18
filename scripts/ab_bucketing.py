"""Interleaved A/B of bucketed vs plain shuffling, through the real loop.

Alternates the two arms so any machine drift hits both equally, and reports
paired ratios. Uses the real fit-only rows at 1024/320 through scripts/train_main
entries rather than a synthetic batch, so the number is the one that ships.

    uv run python scripts/ab_bucketing.py [--reps N]
"""

import argparse
import os
import statistics
import time

for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(name, "1")

import torch

from systemone import ModelConfig, build_model, set_seed
from systemone.bpe import DEFAULT_VOCAB_SIZE, SentencePieceTokenizer
from systemone.corpus import collect_texts, load_train, train_configs
from systemone.data import filter_to_budget, load_or_build
from systemone.paths import CHECKPOINT_DIR
from systemone.train import train

STATE = int(os.environ.get("SYSTEMONE_STATE_TOKENS", "1024"))
QUERY = int(os.environ.get("SYSTEMONE_QUERY_TOKENS", "320"))
MAX_ROWS = int(os.environ.get("SYSTEMONE_MAX_ROWS", "32"))
CONFIGS = int(os.environ.get("BEKKO_CONFIG_LIMIT", "12"))
CASES = int(os.environ.get("BEKKO_CASES", "300"))
NAME = "ab_bucket"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    set_seed(0)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    grouped = load_train(configs=train_configs()[:CONFIGS], max_cases=CASES,
                         max_candidates=8, seed=0)
    examples = [e for case in grouped.values() for e in case]

    directory = CHECKPOINT_DIR / NAME
    tokenizer_file = directory / "tokenizer.model"
    if tokenizer_file.exists():
        tokenizer = SentencePieceTokenizer(tokenizer_file)
    else:
        from systemone.bpe import train_tokenizer
        tokenizer = train_tokenizer(collect_texts(grouped), directory / "tokenizer",
                                   vocab_size=DEFAULT_VOCAB_SIZE)

    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=128, n_layers=4,
                      n_heads=4, n_kv_heads=2, ffn_hidden=352,
                      max_state_tokens=STATE, max_query_tokens=QUERY)
    kept = filter_to_budget(examples, tokenizer, cfg)
    data = load_or_build(directory / "train_cache.pt", kept, tokenizer, cfg)
    print(f"budget={STATE}/{QUERY} rows={MAX_ROWS} examples={len(data):,} "
          f"real_state_tokens={data.state_data.numel():,} device={device.type}")

    model = build_model(cfg, 0)
    arms = {"plain": [], "bucketed": []}
    ratios = []
    for rep in range(args.reps):
        for arm, bucket in (("plain", False), ("bucketed", True)):
            model = build_model(cfg, 0)
            started = time.perf_counter()
            train(model, data, epochs=1, max_rows=MAX_ROWS, lr=2e-3,
                  device=device, seed=0, bf16=True, checkpoint=True,
                  grad_accum=8, bucket=bucket)
            arms[arm].append(time.perf_counter() - started)
        ratios.append(arms["plain"][-1] / arms["bucketed"][-1])
        print(f"rep {rep}: plain={arms['plain'][-1]:.1f}s "
              f"bucketed={arms['bucketed'][-1]:.1f}s ratio={ratios[-1]:.2f}x", flush=True)

    print()
    for arm in ("plain", "bucketed"):
        values = arms[arm]
        print(f"{arm:<9} median={statistics.median(values):6.1f}s "
              f"range={min(values):.1f}-{max(values):.1f}")
    print(f"paired speedup median={statistics.median(ratios):.2f}x "
          f"range={min(ratios):.2f}-{max(ratios):.2f}")


if __name__ == "__main__":
    main()
