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

    def forward(self, x: Tensor, pos: Tensor, mask: Tensor) -> Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = _rope(q, pos, self.rope_base)
        k = _rope(k, pos, self.rope_base)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None], enable_gqa=self.n_heads != self.n_kv_heads
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

    def forward(self, x: Tensor, pos: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), pos, mask)
        h = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


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
        total = s_len + q_len
        device = state_ids.device

        hidden = torch.cat([self.embed(state_ids), self.embed(query_ids)], dim=1)
        pos = torch.cat(
            [torch.arange(s_len, device=device), torch.arange(q_len, device=device)]
        ).unsqueeze(0)

        is_query = torch.cat(
            [
                torch.zeros(s_len, dtype=torch.bool, device=device),
                torch.ones(q_len, dtype=torch.bool, device=device),
            ]
        )
        causal = torch.arange(total, device=device).unsqueeze(0) <= torch.arange(
            total, device=device
        ).unsqueeze(1)
        allowed = (~is_query).unsqueeze(0) | (
            is_query.unsqueeze(0) & is_query.unsqueeze(1) & causal
        )
        key_mask = torch.cat([state_mask, query_mask], dim=1).bool()
        mask = allowed.unsqueeze(0) & key_mask.unsqueeze(1)

        for block in self.blocks:
            hidden = block(hidden, pos, mask)

        hidden = self.norm(hidden)
        query_hidden = hidden[:, s_len:, :]
        last = (query_mask.sum(dim=1).clamp(min=1) - 1).view(-1, 1, 1)
        pooled = query_hidden.gather(1, last.expand(-1, 1, hidden.shape[-1])).squeeze(1)
        return self.truth_head(pooled).squeeze(-1)
