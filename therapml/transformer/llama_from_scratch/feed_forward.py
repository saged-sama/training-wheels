from torch.nn import Module, Linear, Dropout
import torch.nn.functional as F

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = Linear(dim, hidden_dim * 2)
        self.w2 = Linear(hidden_dim, dim)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x_proj = self.w1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        return self.w2(self.dropout(F.silu(x1) * x2))  # SwiGLU