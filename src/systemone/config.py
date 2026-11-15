import functools
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 258
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_hidden: int = 688
    max_state_tokens: int = 192
    max_query_tokens: int = 64
    max_candidates: int = 8
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if min(self.max_state_tokens, self.max_query_tokens, self.max_candidates) <= 0:
            raise ValueError("token and candidate budgets must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @functools.cache
    def parameter_count(self) -> int:
        """Exact parameter count, measured from the constructed module.

        This builds the model on PyTorch's `meta` device, which materialises the
        full parameter graph without allocating any storage, so the number is
        whatever the layers actually contain instead of a hand-maintained formula
        that can drift. (It did drift: the previous formula over-reported by 768
        parameters by counting three norms per block instead of two.)
        """
        from .model import TinyJev

        with torch.device("meta"):
            model = TinyJev(self)
        return sum(parameter.numel() for parameter in model.parameters())

    def kv_cache_bytes(self, seq_len: int) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * seq_len
