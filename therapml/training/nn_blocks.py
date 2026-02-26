import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=None)
    
class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (1 + torch.exp(-x))

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    
class GELU_SIGMOID_AVERAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU() 

    def forward(self, x):
        s = self.sigmoid(x)
        g = self.gelu(x)
        # denom = s + g + 1e-8
        
        return torch.sqrt((s/2.0) * s + (g/2.0) * g)

class SoftMax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_val = torch.max(x, dim=self.dim, keepdim=True).values

        exp_tensor = torch.exp(x-max_val)
        sum_exp = torch.sum(exp_tensor, dim=self.dim, keepdim=True)

        return exp_tensor / sum_exp
    
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        fan_in = in_features * 4
        std = torch.sqrt(torch.tensor(2.0 / fan_in))
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * std)

    def forward(self, x):
        return x @ self.weights.T
    
class BiasLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        fan_in = in_features * 4
        std = torch.sqrt(torch.tensor(2.0 / fan_in))
        self.weights = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weights.T + self.bias
    
class SwiGLU(nn.Module):
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