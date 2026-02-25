import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: Float[Tensor, "batch output_dim"], ground_truth: Float[Tensor, "batch output_dim"]):
        log_probs = F.log_softmax(logits, dim=1)
        target_indices = ground_truth.argmax(dim=1)
        target_log_probs = log_probs[range(logits.shape[0]), target_indices]
        loss = -target_log_probs.mean()
        return loss