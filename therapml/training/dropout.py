import torch.nn as nn
import torch
from jaxtyping import Float
from torch import Tensor

class Dropout(nn.Module):
    def __init__(self, p: float, eps: float = 1e-8):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: Float[Tensor, "..."]):
        if not self.training:
            return x
        
        mask = (torch.rand_like(x) > self.p).float()

        return (x * mask) / (1.0 - self.p + self.eps)