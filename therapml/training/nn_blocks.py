import numpy as np
import torch
# import torch.nn as nn
import math
from torch import Tensor
from jaxtyping import Float
import torch.nn.functional as F

class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=None)

class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))

class SoftMax(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_val = torch.max(x, dim=self.dim, keepdim=True).values

        exp_tensor = torch.exp(x-max_val)
        sum_exp = torch.sum(exp_tensor, dim=self.dim, keepdim=True)

        return exp_tensor / sum_exp

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, weight=None, bias=None):
        super().__init__()
        if weight is None:
            self.weights = torch.nn.Parameter(torch.randn(in_features, out_features))
        else:
            self.weight = weight
        if bias is None:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = bias

    def forward(self, x):
        return x @ self.weight.T + self.bias
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, w1_weight, w2_weight, w3_weight):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = w1_weight
        self.w2_weight = w2_weight
        self.w3_weight = w3_weight
    
    def forward(self, x):
        gate = x @ self.w1_weight.T
        silu = F.silu(gate)
        data = x @ self.w3_weight.T

        swiglu = silu * data
        return swiglu @ self.w2_weight.T