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

    def attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
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
        state_pos: Tensor,
        query_pos: Tensor,
        state_mask: Tensor,
        query_attn_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Two attention calls, no rows x tokens^2 grid. The state half is
        # bidirectional and needs only a per-key flag; the query half needs
        # causality (a per-query-position rule) plus the same flag. Splitting by
        # query set keeps both masks small instead of building one grid per row.
        state_q, state_k, state_v = self.attn.project(self.attn_norm(state), state_pos)
        query_q, query_k, query_v = self.attn.project(self.attn_norm(query), query_pos)

        state_out = self.attn.attend(
            state_q, state_k, state_v, state_mask[:, None, None, :]
        )
        query_out = self.attn.attend(
            query_q,
            torch.cat([state_k, query_k], dim=2),
            torch.cat([state_v, query_v], dim=2),
            query_attn_mask,
        )
        return self._ffn(state + state_out), self._ffn(query + query_out)


def _query_mask(state_mask: Tensor, query_mask: Tensor) -> Tensor:
    """What a query token may see: all real state, and queries up to itself.

    Shape is [rows, 1, query_len, total], not [rows, 1, total, total] -- the
    causal rule depends only on relative position, so it costs query_len * total
    instead of total^2.
    """
    s_len = state_mask.shape[1]
    q_len = query_mask.shape[1]
    total = s_len + q_len
    keys = torch.arange(total, device=state_mask.device)
    queries = torch.arange(q_len, device=state_mask.device)
    allowed = keys.unsqueeze(0) <= (s_len + queries).unsqueeze(1)
    key_mask = torch.cat([state_mask, query_mask], dim=1).bool()
    return (allowed.unsqueeze(0) & key_mask.unsqueeze(1)).unsqueeze(1)


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
    ) -> Tensor:
        s_len = state_ids.shape[1]
        q_len = query_ids.shape[1]
        device = state_ids.device

        state = self.embed(state_ids)
        query = self.embed(query_ids)
        state_pos = torch.arange(s_len, device=device).unsqueeze(0)
        query_pos = torch.arange(q_len, device=device).unsqueeze(0)
        query_attn_mask = _query_mask(state_mask, query_mask)

        for block in self.blocks:
            state, query = block(
                state, query, state_pos, query_pos, state_mask, query_attn_mask
            )

        query = self.norm(query)
        last = (query_mask.sum(dim=1).clamp(min=1) - 1).view(-1, 1, 1)
        pooled = query.gather(1, last.expand(-1, 1, query.shape[-1])).squeeze(1)
        return self.truth_head(pooled).squeeze(-1)

