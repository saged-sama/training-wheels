import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor

class SelfAttention(nn.Module):
    def __init__(self, K: Tensor, V: Tensor, mask):
        super().__init__()
        self.K = K
        self.V = V
        self.mask = mask

    def forward(self, Q):
        
        d_k = self.K.shape[-1]
        scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        scores = (Q @ self.K.mT) / scale
        
        if self.mask is not None:
            scores = scores.masked_fill(~self.mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        output = attention_weights @ self.V
        
        return output
