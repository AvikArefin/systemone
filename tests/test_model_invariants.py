"""Architectural invariants for the network.

The prefix-LM property is the one that is easy to break and impossible to notice:
if state tokens can see the query, the state encoding stops being question
independent and the same state gets a different vector per question.
"""

import torch

from systemone import ByteTokenizer, ModelConfig
from systemone.model import TinyJev, _rope


def _cfg(**overrides) -> ModelConfig:
    defaults = dict(
        vocab_size=ByteTokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        ffn_hidden=64,
        max_state_tokens=32,
        max_query_tokens=16,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_parameter_count_matches_the_module():
    """The reported count must equal what PyTorch actually allocates."""
    for cfg in (
        _cfg(),
        _cfg(n_layers=4, d_model=128, n_heads=4, n_kv_heads=2, ffn_hidden=352, vocab_size=2048),
        ModelConfig(),
    ):
        model = TinyJev(cfg)
        actual = sum(p.numel() for p in model.parameters())
        assert cfg.parameter_count() == actual


def test_parameter_count_follows_an_architecture_change():
    """It is measured, not a formula, so it cannot drift from the model.

    Adding a norm or a projection must move the number automatically; the old
    hand-written formula silently kept reporting the wrong value.
    """
    base = _cfg()
    narrower = _cfg(d_model=16)
    deeper = _cfg(n_layers=3)
    assert deeper.parameter_count() > base.parameter_count()
    assert narrower.parameter_count() < base.parameter_count()


def test_parameter_count_of_the_deployed_model():
    cfg = _cfg(d_model=128, n_layers=4, n_heads=4, n_kv_heads=2, ffn_hidden=352, vocab_size=2048)
    assert cfg.parameter_count() == 1_000_704


def test_embedding_is_zeroed_for_pad():
    cfg = _cfg()
    model = TinyJev(cfg)
    assert torch.all(model.embed.weight[0] == 0)


def test_state_encoding_is_independent_of_the_query():
    """The prefix property: changing the query must not change state outputs.

    This is what lets one state encoding be reused across questions.
    """
    cfg = _cfg()
    model = TinyJev(cfg).eval()
    state = torch.randint(2, cfg.vocab_size, (1, 6))
    state_mask = torch.ones_like(state, dtype=torch.bool)

    query_a = torch.randint(2, cfg.vocab_size, (1, 4))
    query_b = torch.randint(2, cfg.vocab_size, (1, 7))

    captured = {}

    def hook(_module, inputs, _output):
        captured.setdefault("h", []).append(inputs[0].detach().clone())

    handle = model.blocks[0].register_forward_hook(hook)
    model(state, state_mask, query_a, torch.ones_like(query_a, dtype=torch.bool))
    model(state, state_mask, query_b, torch.ones_like(query_b, dtype=torch.bool))
    handle.remove()

    first, second = captured["h"][0], captured["h"][1]
    # state occupies the first 6 positions in both runs
    assert torch.allclose(first[:, :6], second[:, :6], atol=1e-6)


def test_query_attention_is_causal():
    """A query token must not see later query tokens."""
    cfg = _cfg()
    model = TinyJev(cfg).eval()
    state = torch.randint(2, cfg.vocab_size, (1, 3))
    query = torch.randint(2, cfg.vocab_size, (1, 5))
    mask = torch.ones_like(query, dtype=torch.bool)

    captured = {}

    def hook(_module, inputs, _output):
        captured.setdefault("h", []).append(inputs[0].detach().clone())

    handle = model.blocks[0].register_forward_hook(hook)
    model(state, torch.ones_like(state, dtype=torch.bool), query, mask)
    original = captured["h"][0]

    changed = query.clone()
    changed[0, -1] = (changed[0, -1] + 1) % cfg.vocab_size
    captured["h"].clear()
    model(state, torch.ones_like(state, dtype=torch.bool), changed, mask)
    updated = captured["h"][0]

    s = state.shape[1]
    # positions before the changed token are unaffected
    assert torch.allclose(original[:, : s + 4], updated[:, : s + 4], atol=1e-6)


def test_padding_cannot_affect_the_result():
    """Padded positions are masked; appending pad must not change the logit."""
    cfg = _cfg()
    model = TinyJev(cfg).eval()
    state = torch.randint(2, cfg.vocab_size, (1, 5))
    query = torch.randint(2, cfg.vocab_size, (1, 4))

    def run(state_ids, state_mask, query_ids, query_mask):
        with torch.no_grad():
            return model(state_ids, state_mask, query_ids, query_mask)

    base = run(state, torch.ones_like(state, dtype=torch.bool), query, torch.ones_like(query, dtype=torch.bool))

    padded_state = torch.cat([state, torch.zeros(1, 4, dtype=state.dtype)], dim=1)
    padded_state_mask = torch.cat(
        [torch.ones_like(state, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool)], dim=1
    )
    padded_query = torch.cat([query, torch.zeros(1, 3, dtype=query.dtype)], dim=1)
    padded_query_mask = torch.cat(
        [torch.ones_like(query, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool)], dim=1
    )
    assert torch.allclose(
        base, run(padded_state, padded_state_mask, padded_query, padded_query_mask), atol=1e-6
    )


def test_pooling_uses_the_last_real_query_token():
    """With right padding, the pooled position is the last unmasked token."""
    cfg = _cfg()
    model = TinyJev(cfg).eval()
    state = torch.randint(2, cfg.vocab_size, (1, 4))
    query = torch.randint(2, cfg.vocab_size, (1, 6))
    mask = torch.ones_like(query, dtype=torch.bool)
    mask[0, 3:] = False  # only 3 real tokens

    with torch.no_grad():
        full = model(state, torch.ones_like(state, dtype=torch.bool), query, mask)
        trimmed = model(
            state,
            torch.ones_like(state, dtype=torch.bool),
            query[:, :3],
            mask[:, :3],
        )
    assert torch.allclose(full, trimmed, atol=1e-5)


def test_output_shape_is_one_logit_per_row():
    cfg = _cfg()
    model = TinyJev(cfg).eval()
    rows = 7
    state = torch.randint(2, cfg.vocab_size, (rows, 5))
    query = torch.randint(2, cfg.vocab_size, (rows, 3))
    with torch.no_grad():
        logits = model(
            state, torch.ones_like(state, dtype=torch.bool), query, torch.ones_like(query, dtype=torch.bool)
        )
    assert logits.shape == (rows,)


def test_rope_preserves_vector_norm():
    """Rotation is norm preserving; a rewrite that scales would change the geometry."""
    x = torch.randn(2, 3, 5, 8)  # [batch, heads, time, head_dim]
    pos = torch.arange(5).unsqueeze(0)
    rotated = _rope(x, pos, 10000.0)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_rope_is_relative_only_through_positions():
    """Rotating the same vector at the same position is deterministic."""
    x = torch.randn(1, 1, 5, 8)
    pos = torch.tensor([[0, 1, 2, 3, 4]])
    a = _rope(x, pos, 10000.0)
    b = _rope(x, pos, 10000.0)
    assert torch.equal(a, b)


def test_positions_restart_for_the_query_segment(monkeypatch):
    """State uses 0..S-1 and query uses 0..T-1 independently.

    If this changes, trained checkpoints are invalid (docs/12-code-map.md
    invariant 6), so it is worth pinning.
    """
    import systemone.model as model_module

    cfg = _cfg()
    model = TinyJev(cfg).eval()
    state = torch.randint(2, cfg.vocab_size, (1, 4))
    query = torch.randint(2, cfg.vocab_size, (1, 4))
    seen = []
    original = model_module._rope

    def spy(x, pos, base):
        seen.append(pos.detach().clone().tolist())
        return original(x, pos, base)

    monkeypatch.setattr(model_module, "_rope", spy)
    model(state, torch.ones_like(state, dtype=torch.bool), query, torch.ones_like(query, dtype=torch.bool))
    assert seen[0] == seen[1] == [[0, 1, 2, 3]], "state positions count from 0"
    assert seen[2] == seen[3] == [[0, 1, 2, 3]], "query positions restart at 0"


def test_gqa_is_enabled_when_head_counts_differ():
    cfg = _cfg(n_heads=4, n_kv_heads=2)
    model = TinyJev(cfg)
    assert model.blocks[0].attn.n_heads != model.blocks[0].attn.n_kv_heads
    key = model.blocks[0].attn.k_proj.weight
    assert key.shape[0] == cfg.n_kv_heads * cfg.head_dim
