"""Memory and time of one attention call under each mask strategy.

The mask is the only real choice in `model.py`: projections and softmax are
fixed. Each strategy runs in its own process, and the tensor footprint is read
from the MPS allocator, which counts live tensors exactly (peak RSS and
`driver_allocated_memory` both hide MPS buffers behind a shared pool).

    uv run python scripts/measure_attention_memory.py

`rows` is the number that matters: a batch holds examples, and an example
expands into one row per candidate, so 16 examples can be thousands of rows.
"""

import json
import os
import subprocess
import sys
import time

import torch

ROWS = [int(v) for v in os.environ.get("SYSTEMONE_MEM_ROWS", "16,64,128,256").split(",")]
STATE = int(os.environ.get("SYSTEMONE_MEM_STATE", "768"))
QUERY = int(os.environ.get("SYSTEMONE_MEM_QUERY", "128"))
HEADS = int(os.environ.get("SYSTEMONE_MEM_HEADS", "4"))
HEAD_DIM = int(os.environ.get("SYSTEMONE_MEM_HEAD_DIM", "32"))
REPS = int(os.environ.get("SYSTEMONE_MEM_REPS", "5"))
STRATEGIES = ("full_bool", "is_causal", "key_pad", "split", "shared")


def _sdpa(rows: int, strategy: str):
    """Return (mask_bytes_estimate, callable) for one strategy."""
    total = STATE + QUERY
    keep = torch.ones(rows, total, dtype=torch.bool, device="mps")
    keep[:, -QUERY // 4 :] = False  # padding on the tail, as in a real batch
    causal = torch.arange(total, device="mps") <= torch.arange(total, device="mps")[:, None]

    if strategy == "full_bool":
        mask = (causal.unsqueeze(0) & keep.unsqueeze(1)).unsqueeze(1)
        return mask, lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=True
        )

    if strategy == "is_causal":
        return None, lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        )

    if strategy == "key_pad":
        pad = keep[:, None, None, :].bool()
        return pad, lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=pad, enable_gqa=True
        )

    # split: state block sees only its own padding flag; the query block keeps a
    # causal mask of shape [rows, 1, query, total] instead of [rows, 1, total, total]
    state = total - QUERY
    state_pad = keep[:, :state][:, None, None, :]
    query_pad = keep[:, state:][:, None, None, :]
    allowed = (
        torch.arange(total, device="mps").unsqueeze(0)
        <= (state + torch.arange(QUERY, device="mps")).unsqueeze(1)
    )

    if strategy == "shared":
        # what the model does for a length-bucketed batch: no state padding, so
        # the causal rule is one [1, 1, query, total] tensor for every row
        shared = allowed.view(1, 1, QUERY, total)
        return shared, lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
            q[:, :, state:], k, v, attn_mask=shared, enable_gqa=True
        )

    query_mask = (allowed.unsqueeze(0) & keep.unsqueeze(1)).unsqueeze(1)

    def call(q, k, v):
        torch.nn.functional.scaled_dot_product_attention(
            q[:, :, :state], k[:, :, :state], v[:, :, :state],
            attn_mask=state_pad, enable_gqa=True,
        )
        return torch.nn.functional.scaled_dot_product_attention(
            q[:, :, state:], k, v, attn_mask=query_mask, enable_gqa=True
        )

    return query_mask, call


def run(rows: int, strategy: str) -> dict:
    total = STATE + QUERY
    q = torch.randn(rows, HEADS, total, HEAD_DIM, device="mps")
    k = torch.randn(rows, 2, total, HEAD_DIM, device="mps")
    v = torch.randn(rows, 2, total, HEAD_DIM, device="mps")
    torch.mps.synchronize()
    baseline = torch.mps.current_allocated_memory()

    mask, call = _sdpa(rows, strategy)
    torch.mps.synchronize()
    mask_mb = (torch.mps.current_allocated_memory() - baseline) / 1e6

    started = time.perf_counter()
    for _ in range(REPS):
        out = call(q, k, v)
    torch.mps.synchronize()
    ms = (time.perf_counter() - started) * 1000 / REPS
    return {"rows": rows, "strategy": strategy, "mask_mb": round(mask_mb, 1), "ms": round(ms, 1)}


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print(json.dumps(run(int(sys.argv[1]), sys.argv[2])))
        sys.exit(0)

    if not torch.backends.mps.is_available():
        sys.exit("this benchmark needs MPS")

    print(f"state={STATE} query={QUERY} total={STATE + QUERY} heads={HEADS} head_dim={HEAD_DIM}")
    print("mask tensor MB / attention ms, one call")

    for rows in ROWS:
        cells = []
        for strategy in STRATEGIES:
            out = subprocess.run(
                [sys.executable, __file__, str(rows), strategy],
                capture_output=True, text=True, env=os.environ,
            )
            if out.returncode != 0:
                cells.append("crashed")
                continue
            result = json.loads(out.stdout)
            cells.append(f"{result['mask_mb']:.0f}/{result['ms']:.0f}")
        print(f"{rows:>6}  " + "".join(f"{cell:>12}" for cell in cells), flush=True)

    print(f"{'rows':>6}  " + "".join(f"{name:>12}" for name in STRATEGIES))
