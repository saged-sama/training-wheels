import numpy as np

def ReLU(tensor):
    return np.where(tensor > 0, tensor, 0)

def GELU(tensor):
    output = 0.5 * tensor * (1 + np.tanh(np.sqrt(2.0/np.pi) * (tensor + 0.044715 * (tensor**3))))
    return output