import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig
from .tokenizer import PAD_ID


def _rope(x: Tensor, pos: Tensor, base: float) -> Tensor:
    half = x.shape[-1] // 2
    inv = 1.0 / (base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    freqs = pos.float().unsqueeze(-1) * inv
    cos = freqs.cos().unsqueeze(1)
    sin = freqs.sin().unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.rope_base = cfg.rope_base
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def project(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return _rope(q, pos, self.rope_base), _rope(k, pos, self.rope_base), v

    def attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        b, _, t, _ = q.shape
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=self.n_heads != self.n_kv_heads
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def _ffn(self, x: Tensor) -> Tensor:
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def forward(
        self,
        state: Tensor,
        query: Tensor,
        group_ids: Tensor,
        state_pos: Tensor,
        query_pos: Tensor,
        state_attn_mask: Tensor | None,
        query_attn_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        # The state half runs once per group, not once per row: every candidate
        # row of a question carries the same state, and the state cannot see the
        # query, so its hidden states -- and its K/V -- are identical across those
        # rows. Only the query half needs to be per row.
        state_q, state_k, state_v = self.attn.project(self.attn_norm(state), state_pos)
        query_q, query_k, query_v = self.attn.project(self.attn_norm(query), query_pos)

        state_out = self.attn.attend(state_q, state_k, state_v, state_attn_mask)
        query_out = self.attn.attend(
            query_q,
            torch.cat([state_k.index_select(0, group_ids), query_k], dim=2),
            torch.cat([state_v.index_select(0, group_ids), query_v], dim=2),
            query_attn_mask,
        )
        return self._ffn(state + state_out), self._ffn(query + query_out)


def _causal_rule(s_len: int, q_len: int, device: torch.device) -> Tensor:
    """Every query sees all state, then queries up to itself.

    One tensor for the whole batch, `[1, 1, query_len, total]`: the rule depends
    only on relative position, so no row needs its own copy. Worth stating,
    because `is_causal=True` cannot express it -- PyTorch's flag aligns the
    triangle top-left, which for q < keys would let a query see only the first
    few state tokens.
    """
    total = s_len + q_len
    keys = torch.arange(total, device=device).unsqueeze(0)
    queries = torch.arange(q_len, device=device).unsqueeze(1)
    return (keys <= s_len + queries).view(1, 1, q_len, total)


def _query_mask(
    causal: Tensor, state_mask: Tensor, query_mask: Tensor, group_ids: Tensor
) -> Tensor:
    """The shared causal rule, narrowed to the keys each row actually has."""
    key_mask = torch.cat(
        [state_mask.index_select(0, group_ids), query_mask], dim=1
    ).bool()
    return causal & key_mask[:, None, None, :]


class TinyJev(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.truth_head = nn.Linear(cfg.d_model, 1, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def forward(
        self,
        state_ids: Tensor,
        state_mask: Tensor,
        query_ids: Tensor,
        query_mask: Tensor,
        group_ids: Tensor | None = None,
        checkpoint: bool = False,
    ) -> Tensor:
        """Score every row of a batch.

        `state_ids` holds one state per group, not one per row; `group_ids` says
        which group each row belongs to. Pass `group_ids=None` when the caller
        already repeats the state per row.

        `checkpoint` trades time for memory: each block recomputes its
        activations during the backward pass instead of storing them.
        """
        rows = query_ids.shape[0]
        s_len = state_ids.shape[1]
        q_len = query_ids.shape[1]
        device = state_ids.device
        if group_ids is None:
            group_ids = torch.arange(rows, device=device)

        state = self.embed(state_ids)
        query = self.embed(query_ids)
        state_pos = torch.arange(s_len, device=device).unsqueeze(0)
        query_pos = torch.arange(q_len, device=device).unsqueeze(0)

        # One causal rule for the whole batch. When the batch has no state
        # padding -- which length-bucketed batches do not -- nothing has to be
        # narrowed per row, so the mask stays [1, 1, q, total] and the state half
        # needs no mask at all.
        causal = _causal_rule(s_len, q_len, device)
        unpadded = bool(state_mask.all())
        state_attn_mask = None if unpadded else state_mask[:, None, None, :]
        query_attn_mask = (
            causal
            if unpadded
            else _query_mask(causal, state_mask, query_mask, group_ids)
        )

        for block in self.blocks:
            if checkpoint:
                state, query = torch.utils.checkpoint.checkpoint(
                    block,
                    state,
                    query,
                    group_ids,
                    state_pos,
                    query_pos,
                    state_attn_mask,
                    query_attn_mask,
                    use_reentrant=False,
                )
            else:
                state, query = block(
                    state,
                    query,
                    group_ids,
                    state_pos,
                    query_pos,
                    state_attn_mask,
                    query_attn_mask,
                )

        query = self.norm(query)
        last = (query_mask.sum(dim=1).clamp(min=1) - 1).view(-1, 1, 1)
        pooled = query.gather(1, last.expand(-1, 1, query.shape[-1])).squeeze(1)
        return self.truth_head(pooled).squeeze(-1)

