import numpy as np
import torch

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