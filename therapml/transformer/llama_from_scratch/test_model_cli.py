from pathlib import Path
from collections import Counter

import torch

from encoder_decoder import subsequent_mask
from model import make_model
from tokenizer import CharTokenizer


def sample_next_token(
    log_probs,
    generated,
    temperature=0.9,
    top_k=40,
    top_p=0.95,
    repetition_penalty=0.12,
):
    adjusted = log_probs.clone()
    if generated:
        counts = Counter(generated)
        for token_id, count in counts.items():
            adjusted[token_id] -= repetition_penalty * count

    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    adjusted = adjusted / temperature

    if top_k is not None and top_k > 0 and top_k < adjusted.numel():
        topk_vals, topk_idx = torch.topk(adjusted, top_k)
        filtered = torch.full_like(adjusted, float("-inf"))
        filtered[topk_idx] = topk_vals
        adjusted = filtered

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(adjusted, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False
        adjusted[sorted_indices[sorted_mask]] = float("-inf")

    probs = torch.softmax(adjusted, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def sample_generate(
    model,
    prompt_ids,
    pad_idx,
    max_new_tokens,
    device,
    temperature=0.9,
    top_k=40,
    top_p=0.95,
    repetition_penalty=0.12,
):
    generated = prompt_ids[:]
    for _ in range(max_new_tokens):
        src = torch.tensor([generated], dtype=torch.long, device=device)
        seq_mask = (src != pad_idx).unsqueeze(-2)
        causal_mask = subsequent_mask(src.size(1)).type_as(seq_mask.data)
        src_mask = seq_mask & causal_mask
        out = model.forward(src, src, src_mask, src_mask)
        log_probs = model.generator(out[:, -1]).squeeze(0)
        next_token = sample_next_token(
            log_probs,
            generated,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        generated.append(next_token)
    return generated


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stoi = checkpoint["stoi"]
    itos = {int(k): v for k, v in checkpoint["itos"].items()}
    tokenizer = CharTokenizer(stoi=stoi, itos=itos)

    model_config = checkpoint["model_config"]
    model = make_model(
        model_config["src_vocab"],
        model_config["tgt_vocab"],
        N=model_config.get("N", 2),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def run_cli():
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = script_dir / "tiny_shakespeare_transformer.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train first with main.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(checkpoint_path, device)

    print("Tiny Shakespeare CLI ready.")
    print("Type a prompt and press Enter. Type 'exit' to quit.")
    print("Sampling defaults: temperature=0.9, top_k=40, top_p=0.95")

    while True:
        prompt = input("prompt> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        max_new_input = input("max_new_tokens [default=80]> ").strip()
        if not max_new_input:
            max_new_tokens = 80
        else:
            try:
                max_new_tokens = int(max_new_input)
            except ValueError:
                continuation = [max_new_input]
                print(
                    "Detected text instead of max_new_tokens; treating it as a continued prompt."
                )
                print("Finish continuation with an empty line.")
                while True:
                    extra_line = input("... ")
                    if extra_line == "":
                        break
                    continuation.append(extra_line)
                prompt = prompt + "\n" + "\n".join(continuation)
                max_new_tokens = 80

        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        decoded = sample_generate(
            model,
            prompt_ids=prompt_ids,
            pad_idx=tokenizer.pad_idx,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        text = tokenizer.decode(decoded, skip_special_tokens=True)
        print(f"generated> {text}")


if __name__ == "__main__":
    run_cli()
