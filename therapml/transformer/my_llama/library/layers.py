import torch
import torch.nn as nn
from torch import Tensor
from therapml.transformer.my_llama.library.activations import SwiGLU
from therapml.transformer.self_attention import MultiHeadSelfAttentionWithRope
from therapml.training.normalizers import RMSNorm
from therapml.training.nn_blocks import SwiGLU


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor):
        super().__init__()
        # Create linear layers - weights are already in PyTorch format (out_features, in_features)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w1.weight.data = w1_weight
        
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w3.weight.data = w3_weight
        
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w2.weight.data = w2_weight

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_model)
        # Apply w1 and w3 projections: (..., d_model) -> (..., d_ff)
        w1_out = self.w1(x)
        w3_out = self.w3(x)
        
        # SwiGLU: gate * value
        import torch.nn.functional as F
        gate = F.silu(w3_out)
        combined = gate * w1_out
        
        # Apply w2 projection back to d_model: (..., d_ff) -> (..., d_model)
        output = self.w2(combined)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        ctx_len: int = None,
        theta: float = None,
        weights: dict[str, Tensor] = None
    ):
        super().__init__()
        
        # Support both interfaces:
        # 1. For model.py: TransformerBlock(dim, num_heads, dropout)  
        # 2. For test adapter: TransformerBlock(d_model, num_heads, d_ff, ctx_len, theta, weights=...)
        
        if weights is None:
            # Model.py interface: Create trainable blocks
            # If d_ff is actually dropout (float parameter from model.py call)
            if isinstance(d_ff, float) and ctx_len is None:
                dropout = d_ff
                d_ff = d_model * 4  # Standard FFN expansion ratio
            
            self.ln1 = nn.RMSNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, bias=False)
            self.ln2 = nn.RMSNorm(d_model)
            # Simple FFN: w1 -> SiLU -> w2
            self.ffn_w1 = nn.Linear(d_model, d_ff, bias=False)
            self.ffn_w2 = nn.Linear(d_ff, d_model, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.use_rope = False
        else:
            # Test adapter interface: Use provided weights
            q_proj_weight = weights["attn.q_proj.weight"]
            k_proj_weight = weights["attn.k_proj.weight"]
            v_proj_weight = weights["attn.v_proj.weight"]
            o_proj_weight = weights["attn.output_proj.weight"]
            
            self.ln1 = RMSNorm(gamma=weights["ln1.weight"])
            self.ln2 = RMSNorm(gamma=weights["ln2.weight"])
            
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
                w1_weight=weights["ffn.w1.weight"],
                w2_weight=weights["ffn.w2.weight"],
                w3_weight=weights["ffn.w3.weight"]
            )
            self.use_rope = True
        
        self.d_model = d_model

    def forward(self, x: Tensor, token_positions: Tensor = None) -> Tensor:
        if self.use_rope:
            # Test adapter mode: Generate token positions if not provided
            if token_positions is None:
                batch_size, seq_len = x.shape[:2]
                token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            
            # Pre-norm residual for attention with RoPE
            x_norm = self.ln1(x)
            attn_out = self.attn(x_norm, token_positions)
            x = x + attn_out
        else:
            # Model.py mode: Standard attention without RoPE
            x_norm = self.ln1(x)
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
            x = x + attn_out
        
        # Pre-norm residual for feed-forward
        x_norm = self.ln2(x)
        if self.use_rope:
            ffn_out = self.ffn(x_norm)
        else:
            ffn_out = self.ffn_w2(self.dropout(torch.nn.functional.silu(self.ffn_w1(x_norm))))
        x = x + ffn_out
        
        return x
