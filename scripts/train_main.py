import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.distributed as dist

from systemone import EncodedSet, ModelConfig, build_model, checkpoint_path, load_or_build, save_checkpoint, set_seed
from systemone.data import filter_to_budget
from systemone.bpe import DEFAULT_VOCAB_SIZE, SentencePieceTokenizer, train_tokenizer
from systemone.calibrate import (
    accuracy,
    accuracy_by_kind,
    collect_logits,
    collected_loss,
    expected_calibration_error,
    fit_temperatures,
)
from systemone.corpus import (
    collect_texts,
    load_test,
    load_train,
    train_configs,
)
from systemone.train import autocast_dtype, cleanup_distributed, init_distributed, train

SEED = 0
NAME = os.environ.get("SYSTEMONE_NAME", "bekko_tiny")
MAX_CASES = int(os.environ.get("BEKKO_CASES", "1000"))  # per config, not total
CONFIG_LIMIT = int(os.environ.get("BEKKO_CONFIG_LIMIT", "0"))  # 0 = every config
EPOCHS = int(os.environ.get("SYSTEMONE_EPOCHS", "2"))
MAX_ROWS = int(os.environ.get("SYSTEMONE_MAX_ROWS", "256"))
STATE_TOKENS = int(os.environ.get("SYSTEMONE_STATE_TOKENS", "192"))
QUERY_TOKENS = int(os.environ.get("SYSTEMONE_QUERY_TOKENS", "64"))
CHECKPOINT = os.environ.get("SYSTEMONE_CHECKPOINT", "0") == "1"
device, RANK, WORLD_SIZE = init_distributed()
IS_MAIN = RANK == 0
BF16 = os.environ.get("SYSTEMONE_BF16", "1" if device.type == "cuda" else "0") == "1"
COMPILE = os.environ.get("SYSTEMONE_COMPILE", "0") == "1"
GRAD_ACCUM = int(os.environ.get("SYSTEMONE_GRAD_ACCUM", "1"))
RESUME = os.environ.get("SYSTEMONE_RESUME", "0") == "1"
FIT_ONLY = os.environ.get("SYSTEMONE_FIT_ONLY", "0") == "1"
DEV = os.environ.get("SYSTEMONE_DEV", "0") == "1"
INCLUDE_UNFIT = os.environ.get("SYSTEMONE_INCLUDE_UNFIT", "0") == "1"

if DEV:
    # Fast loop: fewer configs, fewer candidates per case, and a smaller data cap.
    # Its own name so it cannot reuse the full run's tokenizer or data cache.
    NAME = f"{NAME}_dev"
    CONFIG_LIMIT = CONFIG_LIMIT or 5
    MAX_CASES = min(MAX_CASES, 200)
    MAX_ROWS = min(MAX_ROWS, 128)
    BF16 = True
    MAX_CANDIDATES = 4
else:
    MAX_CANDIDATES = 8


def _barrier() -> None:
    if WORLD_SIZE > 1:
        dist.barrier()


def main() -> None:
    set_seed(SEED)

    directory = checkpoint_path(NAME)
    tokenizer_file = directory / "tokenizer.model"
    suffix = "_fit" if FIT_ONLY else ""
    train_cache_file = directory / f"bekko_train_cache{suffix}.pt"
    test_cache_file = directory / f"test_cache{suffix}.pt"

    # Only rank 0 touches HuggingFace, trains the tokenizer, and builds caches.
    # Worker ranks wait at the barrier and load the cached tensors in ~5 ms,
    # avoiding duplicate HF downloads / 429 rate-limits on multi-GPU machines.
    test_data = None
    probe_data = None
    if IS_MAIN:
        started = time.perf_counter()
        configs = train_configs(include_unfit=INCLUDE_UNFIT)
        excluded = len(train_configs(include_unfit=True)) - len(configs)
        if CONFIG_LIMIT:
            configs = configs[:CONFIG_LIMIT]
        print(f"loading train split ({len(configs)} configs, up to {MAX_CASES} cases/config)...", flush=True)
        train_grouped = load_train(
            configs=configs, max_candidates=MAX_CANDIDATES, max_cases=MAX_CASES, seed=SEED
        )
        train_set = [example for case in train_grouped.values() for example in case]
        print(f"train load_s={time.perf_counter() - started:.1f} configs={len(configs)} "
              f"excluded_unfit={excluded} cases={len(train_grouped)} examples={len(train_set)}")

        test_grouped = load_test(seed=SEED)
        test_set = [example for case in test_grouped.values() for example in case]

        counts = {"noul": 0, "choice": 0, "score": 0}
        for example in train_set:
            counts[type(example.question).__name__.lower()] += 1
        print(f"train kinds={counts}")
        print(f"test examples={len(test_set)}")

        if tokenizer_file.exists():
            tokenizer = SentencePieceTokenizer(tokenizer_file)
            if tokenizer.vocab_size != DEFAULT_VOCAB_SIZE:
                print(
                    f"warning: reusing a {tokenizer.vocab_size}-piece tokenizer; a new "
                    f"vocabulary is {DEFAULT_VOCAB_SIZE} pieces. Delete {tokenizer_file} "
                    f"or use a new checkpoint name to rebuild it."
                )
            print(f"tokenizer reused vocab={tokenizer.vocab_size}")
        else:
            tokenizer = train_tokenizer(collect_texts(train_grouped), directory / "tokenizer", vocab_size=DEFAULT_VOCAB_SIZE)
            print(f"tokenizer vocab={tokenizer.vocab_size}")

        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            ffn_hidden=352,
            max_state_tokens=STATE_TOKENS,
            max_query_tokens=QUERY_TOKENS,
        )
        if FIT_ONLY:
            kept_train = filter_to_budget(train_set, tokenizer, cfg)
            kept_test = filter_to_budget(test_set, tokenizer, cfg)
            print(f"fit-only {len(kept_train)}/{len(train_set)} train examples "
                  f"({100*len(kept_train)/len(train_set):.1f}%), "
                  f"{len(kept_test)}/{len(test_set)} test ({100*len(kept_test)/len(test_set):.1f}%)")
            train_set, test_set = kept_train, kept_test

        train_data = load_or_build(train_cache_file, train_set, tokenizer, cfg)
        test_data = load_or_build(test_cache_file, test_set, tokenizer, cfg)
        print(
            f"cache train={len(train_data)} test={len(test_data)} "
            f"state_tokens={train_data.state_data.numel():,} "
            f"query_tokens={train_data.query_data.numel():,}"
        )

        # Evaluation is sized to the training set, never larger: measuring on 5x the data
        # the model sees is not more honest, it is just slower. The probe is what we
        # monitor each epoch and is capped at the training size; the whole held-out split
        # is scored exactly once, at the end.
        probe_size = min(len(test_data), len(train_data))
        if probe_size >= len(test_data):
            probe_data = test_data
        else:
            # Spread the probe across configs instead of taking the first N, which would
            # be the first few configs only.
            per_config = max(1, probe_size // max(1, len(test_grouped)))
            probe_examples = [
                example
                for case in test_grouped.values()
                for example in case[:per_config]
            ][:probe_size]
            probe_data = load_or_build(directory / "probe_cache.pt", probe_examples, tokenizer, cfg)
        print(f"eval probe={len(probe_data)} (test {len(test_data)}, train {len(train_data)})")

    _barrier()
    if not IS_MAIN:
        tokenizer = SentencePieceTokenizer(tokenizer_file)
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            ffn_hidden=352,
            max_state_tokens=STATE_TOKENS,
            max_query_tokens=QUERY_TOKENS,
        )
        train_data = EncodedSet.load(train_cache_file)

    model = build_model(cfg, SEED)
    if IS_MAIN:
        print(f"params={cfg.parameter_count():,} ({cfg.parameter_count() / 1e6:.2f}M) "
              f"device={device} world_size={WORLD_SIZE}")
        amp_label = str(autocast_dtype(device)).split(".")[-1] if BF16 else "fp32"
        print(f"budget={STATE_TOKENS}/{QUERY_TOKENS} rows={MAX_ROWS} accum={GRAD_ACCUM} "
              f"checkpoint={CHECKPOINT} amp={amp_label} compile={COMPILE}")

    state_file = directory / "training_state.pt"
    resume = torch.load(state_file, weights_only=False) if (RESUME and state_file.exists()) else None
    if resume and IS_MAIN:
        print(f"resuming from epoch {resume['epoch']} (target {EPOCHS})")

    def save_state(epoch, state):
        torch.save(state, state_file)
        print(f"  saved state at epoch {epoch}")

    started = time.perf_counter()
    history = train(
        model,
        train_data,
        epochs=EPOCHS,
        max_rows=MAX_ROWS,
        lr=2e-3,
        device=device,
        seed=SEED,
        warmup_fraction=0.05,
        eval_dataset=probe_data,
        eval_every=1,
        grad_accum=GRAD_ACCUM,
        checkpoint=CHECKPOINT,
        bf16=BF16,
        resume=resume,
        on_epoch_end=save_state,
        compile=COMPILE,
        progress=True,
    )
    if not IS_MAIN:
        return

    print(f"train_s={time.perf_counter() - started:.1f}")
    print(f"steps={len(history.losses)}")
    print(f"eval_curve={[(e, round(v, 4)) for e, v in history.evals]}")
    print(f"loss first={history.losses[0]:.4f} last={history.losses[-1]:.4f}")

    # One pass over the held-out split produces both the loss and the logits used for
    # accuracy and calibration. Running `evaluate` and then `collect_logits` walked the
    # same examples twice to compute two things from one forward pass.
    collected = collect_logits(model, test_data, device=device, max_rows=MAX_ROWS, bf16=BF16)
    after = collected_loss(collected)
    scope = "fit-only" if FIT_ONLY else "full split"
    print(f"eval_loss={after:.4f} ({scope}, {len(test_data)} examples)")
    print("test_acc", {k: round(v, 3) for k, v in accuracy_by_kind(collected).items()})
    print(f"accuracy={accuracy(collected):.3f}")
    # fit on train: fitting on the eval split would report calibration the model
    # was optimised for, which is leakage
    train_collected = collect_logits(model, train_data, device=device, max_rows=MAX_ROWS, bf16=BF16)
    temperatures = fit_temperatures(train_collected)
    print(f"ece_before={expected_calibration_error(collected):.4f}")
    print("temperatures", {k: round(v, 3) for k, v in temperatures.items()})
    print(f"ece_after={expected_calibration_error(collected, temperatures):.4f}")

    saved = save_checkpoint(
        NAME, model, tokenizer, temperatures=temperatures, seed=SEED,
        extra={"eval_loss": after, "train_cases": MAX_CASES},
    )
    print(f"saved {saved}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
