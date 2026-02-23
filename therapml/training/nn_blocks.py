import numpy as np
import torch
# import torch.nn as nn
import math
from torch import Tensor
from jaxtyping import Float
import torch.nn.functional as F

def ReLU(tensor):
    return np.where(tensor > 0, tensor, 0)

def GELU_tanh(tensor):
    return 0.5 * tensor * (1 + np.tanh(np.sqrt(2.0/np.pi) * (tensor + 0.044715 * (tensor**3))))

def GELU_erf(tensor):
    return 0.5 * tensor * (1 + torch.erf(tensor / torch.sqrt(torch.tensor(2.0))))

def SoftMax(tensor, dim):
    max_val = torch.max(tensor, dim=dim, keepdim=True).values

    exp_tensor = torch.exp(tensor-max_val)
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)

    return exp_tensor / sum_exp

class Linear:
    def __init__(self, in_features, out_features, weight=None):
        self.in_features = in_features
        self.out_features = out_features

        if weight is None:
            self.weight = torch.empty(out_features, in_features)
        else:
            self.weight = weight

    def forward(self, x):
        return x @ self.weight.T

    def __call__(self, x):
        return self.forward(x)
    

class LinearKaiming:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.empty(out_features, in_features)
        self.bias = torch.zeros(out_features)

        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_features
        std = math.sqrt(2.0 / fan_in)
        with torch.no_grad():
            self.weight.normal_(0, std)
            self.bias.zero_()

    def forward(self, x):
        return x @ self.weight.T + self.bias

    def __call__(self, x):
        return self.forward(x)

def SwiGLU(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"]
    ):
    gate = in_features @ w1_weight.T
    silu = F.silu(gate)
    data = in_features @ w3_weight.T

    swiglu = silu * data

    return swiglu @ w2_weight.T