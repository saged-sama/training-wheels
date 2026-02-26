import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

class LayerNorm(nn.Module):
    def __init__(self, gamma: Float[Tensor, "batch ..."], beta: Float[Tensor, "batch ..."], eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
    
    def forward(self, x: Float[Tensor, "batch ..."]):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x_hat + self.beta

class RMSNorm(nn.Module):
    def __init__(self, gamma: Float[Tensor, "batch ..."], eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, x: Float[Tensor, "batch ..."]):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_hat = x / rms

        return self.gamma * x_hat