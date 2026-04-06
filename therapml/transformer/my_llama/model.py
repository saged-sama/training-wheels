import torch
import torch.nn as nn
import torch.nn.functional as F

from therapml.transformer.my_llama.library.layers import TransformerBlock
from therapml.transformer.rope import RoPE


class Llama(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=128,
        num_layers=4,
        num_heads=4,
        block_size=256,
        dropout=0.2,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = RoPE(dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        bsz, seq_len = idx.shape

        token_emb = self.token_embed(idx)
        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.pos_embed(token_emb, positions)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        _, _, channels = logits.shape
        logits = logits.view(bsz * seq_len, channels)
        targets = targets.view(bsz * seq_len)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, eos_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                raise ValueError("temperature must be > 0")
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and torch.all(idx_next == eos_id):
                break

        return idx
