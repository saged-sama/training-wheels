import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from jaxtyping import Float, Int
from .rope import RoPE
import math

class SelfAttention(nn.Module):
    def __init__(self, K: Tensor, V: Tensor, mask):
        super().__init__()
        self.K = K
        self.V = V
        self.mask = mask

    def forward(self, Q):
        d_k = self.K.shape[-1]
        scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        scores = (Q @ self.K.mT) / scale
        
        if self.mask is not None:
            scores = scores.masked_fill(~self.mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        output = attention_weights @ self.V
        
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # Should ideally be nn.Parameter if trainable
        self.q_proj_weight = nn.Parameter(q_proj_weight)
        self.k_proj_weight = nn.Parameter(k_proj_weight)
        self.v_proj_weight = nn.Parameter(v_proj_weight)
        self.o_proj_weight = nn.Parameter(o_proj_weight)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    def forward(self, in_features: torch.Tensor):
        batch, seq_len, _ = in_features.shape

        head_dim = self.d_model // self.num_heads

        # Linear projections
        Q = in_features @ self.q_proj_weight.T
        K = in_features @ self.k_proj_weight.T
        V = in_features @ self.v_proj_weight.T

        # Split heads
        Q = Q.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(head_dim)
        scores = (Q @ K.transpose(-2, -1)) / scale

        # Causal mask (upper triangular)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1
        )

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Attention output
        output = attention_weights @ V

        # Merge heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.d_model)

        return output @ self.o_proj_weight.T
    
class MultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ctx_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, "d_k d_in"],
        k_proj_weight: Float[Tensor, "d_k d_in"],
        v_proj_weight: Float[Tensor, "d_v d_in"],
        o_proj_weight: Float[Tensor, "d_model d_v"],
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads

        self.rope = RoPE(self.head_dim, theta, ctx_len)

        # Should ideally be nn.Parameter if trainable
        self.q_proj_weight = nn.Parameter(q_proj_weight)
        self.k_proj_weight = nn.Parameter(k_proj_weight)
        self.v_proj_weight = nn.Parameter(v_proj_weight)
        self.o_proj_weight = nn.Parameter(o_proj_weight)

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    def forward(self, in_features: torch.Tensor, token_positions: Int[Tensor, "batch ctx_len"]):
        batch, seq_len, _ = in_features.shape

        # Linear projections
        Q = in_features @ self.q_proj_weight.T
        K = in_features @ self.k_proj_weight.T
        V = in_features @ self.v_proj_weight.T

        # Split heads
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        B, H, S, D = Q.shape

        # Reshape for rope application: (B, H, S, D) -> (B*H, S, D)
        Q = Q.reshape(B * H, S, D)
        K = K.reshape(B * H, S, D)

        # Expand positions to match batch size and repeat for heads
        positions = token_positions.expand(B, -1).repeat_interleave(H, dim=0)

        Q = self.rope(Q, positions)
        K = self.rope(K, positions)

        # Reshape back: (B*H, S, D) -> (B, H, S, D)
        Q = Q.reshape(B, H, S, D)
        K = K.reshape(B, H, S, D)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = (Q @ K.transpose(-2, -1)) / scale

        # Causal mask (upper triangular)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
            diagonal=1
        )

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Attention output
        output = attention_weights @ V

        # Merge heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.d_model)

        return output @ self.o_proj_weight.T
    