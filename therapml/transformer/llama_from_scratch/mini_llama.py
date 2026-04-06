from torch.nn import Module, ModuleList, Parameter, Embedding, RMSNorm, Linear
import torch

try:
    from .transformer_block import TransformerBlock
except ImportError:
    from transformer_block import TransformerBlock

class Llama(Module):
    def __init__(self, vocab_size, dim, ff_hidden_dim, n_layers, heads=8, ctx_len=4096, theta=10000.0):
        super().__init__()
        self.token_emb = Embedding(vocab_size, dim)
        self.pos_emb = Parameter(torch.zeros(1, ctx_len, dim))
        self.blocks = ModuleList(
            [TransformerBlock(dim, ff_hidden_dim, heads, ctx_len=ctx_len, theta=theta) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)
        self.output = Linear(dim, vocab_size, bias=False)
        self.output.weight = self.token_emb.weight
        self.ctx_len = ctx_len

    def forward(self, x):
        batch, seq_len = x.shape
        if seq_len > self.ctx_len:
            raise ValueError(f"sequence length {seq_len} exceeds ctx_len {self.ctx_len}")

        token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(x)
        for block in self.blocks:
            x = block(x, token_positions)
        x = self.norm(x)
        return self.output(x)
    
