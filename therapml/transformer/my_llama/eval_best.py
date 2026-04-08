import argparse
import json
from pathlib import Path

import torch

from therapml.transformer.my_llama.library.train_utils import estimate_loss
from therapml.transformer.my_llama.model import Llama


BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
TOKENIZER_META_PATH = LOGS_DIR / "tokenizer_config.json"
TRAIN_PATH = BASE_DIR / "tiny_stories" / "train.json"
VAL_PATH = BASE_DIR / "tiny_stories" / "val.json"

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
EVAL_CONFIG = {
    "batch_size": 4,
    "eval_iters": 200,
}


class _SentencePieceWrapper:
    def __init__(self, processor):
        self.processor = processor

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.processor.decode(ids)

    def bos_id(self) -> int:
        return self.processor.bos_id() if self.processor.bos_id() >= 0 else self.processor.unk_id()

    def eos_id(self) -> int | None:
        return self.processor.eos_id() if self.processor.eos_id() >= 0 else None


class _HFBPEWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def bos_id(self) -> int:
        bos = self.tokenizer.token_to_id("<s>")
        unk = self.tokenizer.token_to_id("<unk>")
        if bos is None and unk is None:
            raise ValueError("Tokenizer is missing both <s> and <unk> special tokens")
        return bos if bos is not None else unk

    def eos_id(self) -> int | None:
        return self.tokenizer.token_to_id("</s>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare train-best and eval-best Mini-Llama checkpoints."
    )
    parser.add_argument(
        "--train-checkpoint",
        type=Path,
        default=LOGS_DIR / "mini_llama_best_train.pt",
        help="Path to the best train checkpoint",
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=LOGS_DIR / "mini_llama_best_eval.pt",
        help="Path to the best eval checkpoint",
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text for generation")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--eval-iters", type=int, default=EVAL_CONFIG["eval_iters"], help="Number of eval batches")
    parser.add_argument("--batch-size", type=int, default=EVAL_CONFIG["batch_size"], help="Batch size for loss eval")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1337,
        help="Random seed used before each loss eval for fair comparison",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=2024,
        help="Random seed used before each generation for fair comparison",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu or cuda")
    return parser.parse_args()


def load_text_data() -> tuple[str, str]:
    with TRAIN_PATH.open("r", encoding="utf-8") as file:
        train_json = json.load(file)

    with VAL_PATH.open("r", encoding="utf-8") as file:
        val_json = json.load(file)

    train_text = "\n\n".join(row["row"]["text"] for row in train_json["rows"])
    val_text = "\n\n".join(row["row"]["text"] for row in val_json["rows"])
    return train_text, val_text


def encode_text(tokenizer, text: str, bos_id: int | None, eos_id: int | None) -> torch.Tensor:
    ids = tokenizer.encode(text)
    if bos_id is not None:
        ids = [bos_id] + ids
    if eos_id is not None:
        ids = ids + [eos_id]
    return torch.tensor(ids, dtype=torch.long)


def build_model(device: str) -> Llama:
    return Llama(
        vocab_size=VOCAB_SIZE,
        context_length=MODEL_CONFIG["context_length"],
        d_model=MODEL_CONFIG["d_model"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        d_ff=MODEL_CONFIG["d_ff"],
        rope_theta=MODEL_CONFIG["rope_theta"],
        weights=None,
    ).to(device)


def evaluate_checkpoint(
    checkpoint_path: Path,
    tokenizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    args: argparse.Namespace,
    device: str,
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    torch.manual_seed(args.eval_seed)
    losses = estimate_loss(
        model,
        train_data,
        val_data,
        args.eval_iters,
        args.batch_size,
        MODEL_CONFIG["context_length"],
        device,
    )

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id()]

    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    torch.manual_seed(args.sample_seed)
    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_id=tokenizer.eos_id(),
        )[0].tolist()

    return {
        "train_loss": losses["train"].item(),
        "val_loss": losses["val"].item(),
        "generated_text": tokenizer.decode(generated_ids),
    }


def print_report(train_result: dict, eval_result: dict) -> None:
    print("=== Checkpoint Comparison ===")
    print("metric                train-best       eval-best")
    print("--------------------------------------------------")
    print(f"train_loss            {train_result['train_loss']:.6f}      {eval_result['train_loss']:.6f}")
    print(f"val_loss              {train_result['val_loss']:.6f}      {eval_result['val_loss']:.6f}")
    print()
    print("=== train-best generation ===")
    print(train_result["generated_text"])
    print()
    print("=== eval-best generation ===")
    print(eval_result["generated_text"])


def main() -> None:
    args = parse_args()

    if not TOKENIZER_META_PATH.exists():
        raise FileNotFoundError(f"Tokenizer metadata not found: {TOKENIZER_META_PATH}")
    tokenizer_meta = json.loads(TOKENIZER_META_PATH.read_text(encoding="utf-8"))
    tokenizer_path = LOGS_DIR / tokenizer_meta["tokenizer_file"]

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    backend = tokenizer_meta.get("backend", "sentencepiece")
    if backend == "sentencepiece":
        import sentencepiece as spm

        tokenizer = _SentencePieceWrapper(spm.SentencePieceProcessor(model_file=str(tokenizer_path)))
    elif backend == "hf_bpe":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "HF BPE checkpoint requires `tokenizers`. Install with: pip install tokenizers"
            ) from exc
        tokenizer = _HFBPEWrapper(Tokenizer.from_file(str(tokenizer_path)))
    else:
        raise ValueError(f"Unknown tokenizer backend in metadata: {backend}")

    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()

    train_text, val_text = load_text_data()
    train_data = encode_text(tokenizer, train_text, bos_id, eos_id)
    val_data = encode_text(tokenizer, val_text, bos_id, eos_id)

    train_result = evaluate_checkpoint(
        args.train_checkpoint,
        tokenizer,
        train_data,
        val_data,
        args,
        device,
    )
    eval_result = evaluate_checkpoint(
        args.eval_checkpoint,
        tokenizer,
        train_data,
        val_data,
        args,
        device,
    )

    print_report(train_result, eval_result)


if __name__ == "__main__":
    main()
