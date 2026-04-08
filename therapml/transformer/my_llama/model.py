import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int

from therapml.transformer.my_llama.library.layers import TransformerBlock
from therapml.training.normalizers import RMSNorm
from therapml.training.loss import CrossEntropyLoss

class Llama(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int | None = None,
        d_model: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        d_ff: int | None = None,
        rope_theta: float = 10000.0,
        weights: dict[str, Tensor] | None = None,
    ):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, d_model)
        if weights is not None and "token_embeddings.weight" in weights:
            self.token_embed.weight.data.copy_(weights["token_embeddings.weight"])

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.blocks = nn.ModuleList(
            [TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                ctx_len=context_length,
                theta=rope_theta,
                weights={
                    key[len(f"layers.{i}.") :]: value
                    for key, value in (weights or {}).items()
                    if key.startswith(f"layers.{i}.")
                },
            ) for i in range(num_layers)]
        )

        final_gamma = weights.get("ln_final.weight", torch.ones(d_model)) if weights is not None else torch.ones(d_model)
        self.ln_final = RMSNorm(gamma=final_gamma, eps=1e-5)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if weights is not None and "lm_head.weight" in weights:
            self.lm_head.weight.data.copy_(weights["lm_head.weight"])

        self.cross_entropy = CrossEntropyLoss()

    def forward(
        self,
        idx: Int[Tensor, "batch_size sequence_length"],
        targets: Int[Tensor, "batch_size sequence_length"] | None = None,
    ) -> Float[Tensor, "batch_size sequence_length vocab_size"] | tuple[Float[Tensor, "batch_size_sequence_length vocab_size"], Float[Tensor, ""]]:
        idx = idx[:, -self.context_length :]
        if targets is not None:
            targets = targets[:, -self.context_length :]

        x = self.token_embed(idx)
        token_positions = torch.arange(x.size(1), device=idx.device, dtype=torch.long).unsqueeze(0).expand(idx.size(0), -1)

        for block in self.blocks:
            x = block(x, token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        targets_one_hot = F.one_hot(targets_flat, num_classes=logits_flat.size(-1)).to(logits_flat.dtype)
        loss = self.cross_entropy(logits_flat, targets_one_hot)
        return logits_flat, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40, eos_id=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.context_length :])
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                cutoff = top_values[:, [-1]]
                logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

            if eos_id is not None and torch.all(next_idx.squeeze(-1) == eos_id):
                break

        return idx
