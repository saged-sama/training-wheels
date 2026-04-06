import torch

from therapml.transformer.my_llama.model import Llama


def test_llama_forward_shapes():
    model = Llama(vocab_size=64, dim=32, num_layers=2, num_heads=4, block_size=16, dropout=0.1)
    idx = torch.randint(0, 64, (2, 16))
    logits, loss = model(idx, idx)

    assert logits.shape == (32, 64)
    assert loss is not None


def test_llama_generate_extends_sequence():
    model = Llama(vocab_size=32, dim=32, num_layers=1, num_heads=4, block_size=8, dropout=0.1)
    idx = torch.randint(0, 32, (1, 4))
    out = model.generate(idx, max_new_tokens=5, temperature=1.0, top_k=8)

    assert out.shape[1] == 9
