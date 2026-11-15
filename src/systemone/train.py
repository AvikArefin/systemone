import math
import random
from dataclasses import dataclass, field

import torch

from .data import EncodedSet, batch_loss
from .model import TinyJev


@dataclass
class History:
    losses: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)
    evals: list[tuple[int, float]] = field(default_factory=list)


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


def evaluate(
    model: TinyJev,
    dataset: EncodedSet,
    max_rows: int = 256,
    device: torch.device | None = None,
) -> float:
    device = device or next(model.parameters()).device
    model.to(device).eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for chunk in dataset.batches(max_rows):
            batch = dataset.collate(chunk).to(device)
            logits = model(batch.state_ids, batch.state_mask, batch.query_ids, batch.query_mask)
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
) -> History:
    device = device or next(model.parameters()).device
    model.to(device).train()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    steps_per_epoch = len(dataset.batches(max_rows))
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

    for epoch in range(epochs):
        rng.shuffle(order)
        for chunk in _chunks(dataset, order, max_rows):
            batch = dataset.collate(chunk).to(device)
            logits = model(batch.state_ids, batch.state_mask, batch.query_ids, batch.query_mask)
            loss = batch_loss(logits, batch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            scheduler.step()
            history.losses.append(loss.item())
            history.lrs.append(scheduler.get_last_lr()[0])
            step += 1
        if eval_dataset is not None and (epoch + 1) % eval_every == 0:
            history.evals.append((epoch + 1, evaluate(model, eval_dataset, max_rows, device)))
            model.to(device).train()

    model.eval()
    return history
