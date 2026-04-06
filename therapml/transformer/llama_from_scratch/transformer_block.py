from torch.nn import Module, RMSNorm
from therapml.transformer.self_attention import MultiHeadSelfAttentionWithRope

try:
    from .feed_forward import FeedForward
except ImportError:
    from feed_forward import FeedForward

class TransformerBlock(Module):
    def __init__(self, dim, ff_hidden_dim, heads=8, ctx_len=4096, theta=10000.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttentionWithRope(
            d_model=dim,
            num_heads=heads,
            ctx_len=ctx_len,
            theta=theta,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            o_proj_weight=None,
        )
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, ff_hidden_dim)

    def forward(self, x, token_positions):
        x = x + self.attn(self.norm1(x), token_positions)  # Pre-norm
        x = x + self.ff(self.norm2(x))
        return x
