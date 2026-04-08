import argparse
from pathlib import Path

import sentencepiece as spm
import torch

from therapml.transformer.my_llama.model import Llama


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"

# Keep these aligned with train.py defaults.
MODEL_CONFIG = {
    "context_length": 256,
    "d_model": 128,
    "num_layers": 4,
    "num_heads": 4,
    "d_ff": 512,
    "rope_theta": 10000.0,
}
VOCAB_SIZE = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from the best trained Mini-Llama checkpoint.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text to start generation")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = LOGS_DIR / "mini_llama_best.pt"
    tokenizer_path = LOGS_DIR / "spm.model"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    bos_id = tokenizer.bos_id() if tokenizer.bos_id() >= 0 else tokenizer.unk_id()
    eos_id = tokenizer.eos_id() if tokenizer.eos_id() >= 0 else None

    model = Llama(
        vocab_size=VOCAB_SIZE,
        context_length=MODEL_CONFIG["context_length"],
        d_model=MODEL_CONFIG["d_model"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        d_ff=MODEL_CONFIG["d_ff"],
        rope_theta=MODEL_CONFIG["rope_theta"],
        weights=None,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt, out_type=int)
    if not prompt_ids:
        prompt_ids = [bos_id]

    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=eos_id,
        )[0].tolist()

    generated_text = tokenizer.decode(generated_ids)
    print(generated_text)


if __name__ == "__main__":
    main()
