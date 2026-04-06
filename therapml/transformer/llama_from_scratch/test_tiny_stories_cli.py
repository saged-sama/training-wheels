from pathlib import Path

import torch

from encoder_decoder import subsequent_mask
from model import make_model
from tokenizer import CharTokenizer
from test_model_cli import load_checkpoint, sample_generate


def run_cli():
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = script_dir / "tiny_stories_transformer_interrupted.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Train first with: python main.py  (uses LLAMA_QUICK=1 for a quick run)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_checkpoint(checkpoint_path, device)

    print("Tiny Stories CLI ready.")
    print("Type a story opening and press Enter. Type 'exit' to quit.")
    print("Sampling: temperature=0.9, top_k=40, top_p=0.95")

    while True:
        prompt = input("prompt> ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break

        max_new_input = input("max_new_tokens [default=120]> ").strip()
        if not max_new_input:
            max_new_tokens = 120
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
                max_new_tokens = 120

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
