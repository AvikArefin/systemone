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

    def state(self, model, optimiser, scheduler) -> dict:
        """Everything needed to continue this run later."""
        return {
            "epoch": self.epoch,
            "losses": list(self.losses),
            "lrs": list(self.lrs),
            "evals": list(self.evals),
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
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


def _precision(device: torch.device, bf16: bool):
    if not bf16:
        return nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)


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
        for chunk in dataset.batches(max_rows):
            batch = dataset.collate(chunk).to(device)
            with _precision(device, bf16):
                logits = model(
                    batch.state_ids,
                    batch.state_mask,
                    batch.query_ids,
                    batch.query_mask,
                    batch.group_ids,
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
    resume: dict | None = None,
    on_epoch_end: Callable[[int, dict], None] | None = None,
) -> History:
    """Train, optionally continuing a previous run.

    `epochs` is the target total, not a count of additional epochs: resuming at
    epoch 2 with `epochs=6` runs epochs 3-6 and keeps one continuous schedule.

    `resume` is the dict returned in `History.state()` and `on_epoch_end`
    receives a fresh one after each epoch, so the caller decides where it lives.
    """
    device = device or next(model.parameters()).device
    model.to(device).train()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    steps_per_epoch = math.ceil(len(dataset.batches(max_rows)) / grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_fraction)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)
    order = list(range(len(dataset)))
    rng = random.Random(seed)
    history = History()
    step = 0
    start_epoch = 0

    if resume:
        model.load_state_dict(resume["model"])
        start_epoch = int(resume["epoch"])
        history.epoch = start_epoch
        history.losses = list(resume["losses"])
        history.lrs = list(resume["lrs"])
        history.evals = list(resume["evals"])
        optimiser.load_state_dict(resume["optimiser"])
        scheduler.load_state_dict(resume["scheduler"])

    for epoch in range(start_epoch, epochs):
        rng.shuffle(order)
        chunks = list(_chunks(dataset, order, max_rows))
        for start in range(0, len(chunks), grad_accum):
            window = chunks[start : start + grad_accum]
            optimiser.zero_grad(set_to_none=True)
            running = 0.0
            for chunk in window:
                batch = dataset.collate(chunk).to(device)
                with _precision(device, bf16):
                    logits = model(
                        batch.state_ids,
                        batch.state_mask,
                        batch.query_ids,
                        batch.query_mask,
                        batch.group_ids,
                        checkpoint,
                    )
                    loss = batch_loss(logits, batch) / len(window)
                loss.backward()
                running += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            scheduler.step()
            history.losses.append(running)
            history.lrs.append(scheduler.get_last_lr()[0])
            step += 1
        if eval_dataset is not None and (epoch + 1) % eval_every == 0:
            history.evals.append(
                (epoch + 1, evaluate(model, eval_dataset, max_rows, device, bf16))
            )
            model.to(device).train()
        history.epoch = epoch + 1
        if on_epoch_end is not None:
            on_epoch_end(epoch + 1, history.state(model, optimiser, scheduler))

    model.eval()
    return history
