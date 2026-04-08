import torch
import torch.nn as nn
from torch import Tensor
from therapml.transformer.self_attention import MultiHeadSelfAttentionWithRope
from therapml.training.normalizers import RMSNorm
from therapml.training.nn_blocks import SwiGLU


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor):
        super().__init__()
        self.swiglu = SwiGLU(
            d_model=d_model, 
            d_ff=d_ff, 
            w1_weight=w1_weight, 
            w2_weight=w2_weight, 
            w3_weight=w3_weight
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.swiglu(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        ctx_len: int = None,
        theta: float = None,
        weights: dict[str, Tensor] = None
    ):
        super().__init__()
        weights = weights or {}
        q_proj_weight = weights.get("attn.q_proj.weight")
        k_proj_weight = weights.get("attn.k_proj.weight")
        v_proj_weight = weights.get("attn.v_proj.weight")
        o_proj_weight = weights.get("attn.output_proj.weight")
        
        self.ln1 = RMSNorm(gamma=weights.get("ln1.weight", torch.ones(d_model)), eps=1e-5)
        self.ln2 = RMSNorm(gamma=weights.get("ln2.weight", torch.ones(d_model)), eps=1e-5)
        
        self.attn = MultiHeadSelfAttentionWithRope(
            d_model=d_model,
            num_heads=num_heads,
            ctx_len=ctx_len,
            theta=theta,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            o_proj_weight=o_proj_weight
        )
        
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            w1_weight=weights.get("ffn.w1.weight", torch.randn(d_ff, d_model) / (d_model ** 0.5)),
            w2_weight=weights.get("ffn.w2.weight", torch.randn(d_model, d_ff) / (d_ff ** 0.5)),
            w3_weight=weights.get("ffn.w3.weight", torch.randn(d_ff, d_model) / (d_model ** 0.5))
        )
        
        self.d_model = d_model

    def forward(self, x: Tensor, token_positions: Tensor = None) -> Tensor:
        if token_positions is None:
            batch_size, seq_len = x.shape[:2]
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, token_positions)
        x = x + attn_out
            
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        
        return x
