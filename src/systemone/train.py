"""Training loop.

Four memory knobs, all measured in [Experiments](../docs/11-experiments.md):

- `max_rows` caps rows per chunk (a row is one candidate, not one example)
- `grad_accum` sums gradients over several chunks before stepping -- we measured
  this as throughput-free, so it is the cheapest way to buy headroom
- `checkpoint` recomputes block activations in the backward pass: -46% memory,
  +30% time
- `bf16` runs the forward/backward in bfloat16, which saves no memory here but
  removes the fp32 intermediate pressure that makes long sequences swap
"""

import math
import random
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from .data import EncodedSet, batch_loss
from .model import TinyJev


@dataclass
class History:
    losses: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    evals: list[tuple[int, float]] = field(default_factory=list)
    epoch: int = 0

    def state(self, model, optimiser, scheduler, fingerprint: str = "") -> dict:
        """Everything needed to continue this run later.

        `fingerprint` identifies the tokenizer the rows were encoded with. Resume
        compares it: the weights are only meaningful against the vocabulary that
        produced the token ids, and `load_state_dict` cannot tell them apart
        because a same-sized vocabulary is shape-compatible.
        """
        return {
            "epoch": self.epoch,
            "losses": list(self.losses),
            "lrs": list(self.lrs),
            "evals": list(self.evals),
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "tokenizer_fingerprint": fingerprint,
        }


def _chunks(dataset: EncodedSet, order: list[int], max_rows: int):
    counts = dataset.row_counts()
    batch: list[int] = []
    total = 0
    for index in order:
        count = counts[index]
        if batch and total + count > max_rows:
            yield batch
            batch, total = [], 0
        batch.append(index)
        total += count
    if batch:
        yield batch


def default_device() -> torch.device:
    """Pick the fastest available accelerator: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Best reduced-precision dtype for the device.

    CUDA GPUs without native bf16 (e.g. Colab/Kaggle T4, P100) fall back to
    fp16 so they hit Tensor Cores instead of slow bf16 emulation.
    """
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def _precision(device: torch.device, bf16: bool):
    if not bf16:
        return nullcontext()
    return torch.autocast(device.type, dtype=autocast_dtype(device))


def evaluate(
    model: TinyJev,
    dataset: EncodedSet,
    max_rows: int = 256,
    device: torch.device | None = None,
    bf16: bool = False,
) -> float:
    device = device or next(model.parameters()).device
    model.to(device).eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for chunk in dataset.batches(max_rows, by_length=True):
            batch = dataset.collate(chunk).to(device)
            with _precision(device, bf16):
                logits = model(
                    batch.state_ids,
                    batch.state_mask,
                    batch.query_ids,
                    batch.query_mask,
                    batch.group_ids,
                    state_unpadded=batch.state_unpadded,
                )
            total += batch_loss(logits, batch).item()
            count += 1
    return total / count if count else float("nan")


def train(
    model: TinyJev,
    dataset: EncodedSet,
    epochs: int = 20,
    max_rows: int = 256,
    lr: float = 3e-3,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    device: torch.device | None = None,
    seed: int = 0,
    warmup_fraction: float = 0.05,
    min_lr_ratio: float = 0.05,
    eval_dataset: EncodedSet | None = None,
    eval_every: int = 1,
    grad_accum: int = 1,
    checkpoint: bool = False,
    bf16: bool = False,
    bucket: bool = True,
    resume: dict | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
    compile: bool = False,
) -> History:
    """Train, optionally continuing a previous run.

    `epochs` is the target total, not a count of additional epochs: resuming at
    epoch 2 with `epochs=6` runs epochs 3-6 and keeps one continuous schedule.

    `bucket` visits each epoch in length bands, which changes only padding, never
    the loss or which examples share a batch window over a full run. Kept
    switchable so the measurement in [Experiments](../docs/11-experiments.md)
    can be reproduced per epoch.

    `resume` is the dict returned in `History.state()` and `on_epoch_end`
    receives a fresh one after each epoch, so the caller decides where it lives.
    """
    device = device or next(model.parameters()).device
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model.to(device).train()
    runner = torch.compile(model) if compile and device.type == "cuda" else model
    use_scaler = device.type == "cuda" and bf16 and autocast_dtype(device) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Count chunks the same way the loop will build them, or the schedule is
    # sized for a different number of steps than the epoch actually takes.
    counted = (
        dataset.bucketed_order(random.Random(seed))
        if bucket
        else list(range(len(dataset)))
    )
    chunks_per_epoch = len(list(_chunks(dataset, counted, max_rows)))
    steps_per_epoch = math.ceil(chunks_per_epoch / grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_fraction)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)
    rng = random.Random(seed)
    history = History()
    step = 0
    start_epoch = 0

    if resume:
        saved = resume.get("tokenizer_fingerprint")
        current = getattr(dataset, "fingerprint", "")
        if saved is None:
            raise ValueError(
                "cannot resume: this training state predates the tokenizer "
                "fingerprint guard, so its vocabulary cannot be verified. "
                "Start a fresh run instead."
            )
        if saved != current:
            raise ValueError(
                f"cannot resume: the saved state was trained with tokenizer "
                f"{saved}, but these rows were encoded with {current}. The weights "
                "and the token ids mean different things; retrain from scratch."
            )
        model.load_state_dict(resume["model"])
        start_epoch = int(resume["epoch"])
        history.epoch = start_epoch
        history.losses = list(resume["losses"])
        history.lrs = list(resume["lrs"])
        history.evals = list(resume["evals"])
        optimiser.load_state_dict(resume["optimiser"])
        scheduler.load_state_dict(resume["scheduler"])

    for epoch in range(start_epoch, epochs):
        order = (
            dataset.bucketed_order(rng)
            if bucket
            else list(range(len(dataset)))
        )
        if not bucket:
            rng.shuffle(order)
        chunks = list(_chunks(dataset, order, max_rows))
        for start in range(0, len(chunks), grad_accum):
            window = chunks[start : start + grad_accum]
            optimiser.zero_grad(set_to_none=True)
            running = torch.zeros((), device=device)
            for chunk in window:
                batch = dataset.collate(chunk).to(device)
                with _precision(device, bf16):
                    logits = runner(
                        batch.state_ids,
                        batch.state_mask,
                        batch.query_ids,
                        batch.query_mask,
                        batch.group_ids,
                        checkpoint,
                        batch.state_unpadded,
                    )
                    loss = batch_loss(logits, batch) / len(window)
                scaler.scale(loss).backward()
                running = running + loss.detach()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            history.losses.append(running.item())
            history.lrs.append(scheduler.get_last_lr()[0])
            step += 1
        if eval_dataset is not None and (epoch + 1) % eval_every == 0:
            history.evals.append(
                (epoch + 1, evaluate(model, eval_dataset, max_rows, device, bf16))
            )
            model.to(device).train()
        history.epoch = epoch + 1
        if on_epoch_end is not None:
            on_epoch_end(
                epoch + 1,
                history.state(
                    model, optimiser, scheduler,
                    fingerprint=getattr(dataset, "fingerprint", ""),
                ),
            )

    model.eval()
    return history
