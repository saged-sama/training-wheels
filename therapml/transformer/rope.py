from jaxtyping import Float, Int
import torch
from torch import Tensor
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        theta: float = 10000.0,
        context_len: int = 4096,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        powers = torch.arange(0, embedding_dim, 2).float() / embedding_dim
        
        inv_freq = 1.0 / (theta ** powers)
        
        t = torch.arange(context_len).float()
        
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        stack = torch.stack((-x2, x1), dim=-1)
        return stack.flatten(-2)

    def forward(
        self, 
        input_embeddings: Float[Tensor, "batch seq_len embedding_dim"], 
        token_positions: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len embedding_dim"]:
        
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        
        rotated_embeddings = (input_embeddings * cos) + (self._rotate_half(input_embeddings) * sin)
        
        return rotated_embeddings