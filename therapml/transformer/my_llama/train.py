import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from therapml.transformer.my_llama.library.train_utils import estimate_loss, get_batch, get_lr
from therapml.transformer.my_llama.model import Llama
from therapml.training.adam import AdamW

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
TRAIN_PATH = BASE_DIR / "tiny_stories" / "train.json"
VAL_PATH = BASE_DIR / "tiny_stories" / "val.json"
TOKENIZER_META_PATH = LOGS_DIR / "tokenizer_config.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOKENIZER_CONFIG = {
    # Options: "hf_bpe", "sentencepiece"
    "backend": "hf_bpe",
    "vocab_size": 2000,
}

MODEL_CONFIG = {
    "context_length": 256,
    "d_model": 128,
    "num_layers": 4,
    "num_heads": 4,
    "d_ff": 512,
    "rope_theta": 10000.0,
}

TRAIN_CONFIG = {
    "batch_size": 4,
    "learning_rate": 3e-5,
    "min_lr": 3e-8,
    "warmup_iters": 200,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "max_iters": 5000,
    "eval_interval": 100,
    "eval_iters": 200,
    "use_early_stopping": False,
    "patience": 8,
    "min_delta": 1e-3,
    "vocab_size": 2000,
    "seed": 1337,
}

SAMPLING_CONFIG = {
    "max_new_tokens": 400,
    "temperature": 0.8,
    "top_k": 40,
}

SAMPLE_CONFIG = {
    "interval": 500,
    "max_new_tokens": 100,
}


def load_text_data():
    with TRAIN_PATH.open("r", encoding="utf-8") as file:
        train_json = json.load(file)

    with VAL_PATH.open("r", encoding="utf-8") as file:
        val_json = json.load(file)

    train_text = "\n\n".join(row["row"]["text"] for row in train_json["rows"])
    val_text = "\n\n".join(row["row"]["text"] for row in val_json["rows"])
    return train_text, val_text


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


def build_tokenizer_and_tensors(train_text, val_text, vocab_size):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    train_txt_path = LOGS_DIR / "train.txt"
    val_txt_path = LOGS_DIR / "val.txt"

    with train_txt_path.open("w", encoding="utf-8") as file:
        file.write(train_text)

    with val_txt_path.open("w", encoding="utf-8") as file:
        file.write(val_text)

    backend = TOKENIZER_CONFIG["backend"]
    if backend == "sentencepiece":
        import sentencepiece as spm

        spm_prefix = LOGS_DIR / "spm"
        spm.SentencePieceTrainer.Train(
            input=str(train_txt_path),
            model_prefix=str(spm_prefix),
            vocab_size=vocab_size,
            model_type="bpe",
        )

        processor = spm.SentencePieceProcessor()
        processor.Load(str(spm_prefix) + ".model")
        tokenizer = _SentencePieceWrapper(processor)
    elif backend == "hf_bpe":
        try:
            from tokenizers import Tokenizer
            from tokenizers import decoders
            from tokenizers import models
            from tokenizers import pre_tokenizers
            from tokenizers import trainers
        except ImportError as exc:
            raise ImportError(
                "HF BPE backend requires `tokenizers`. Install with: pip install tokenizers"
            ) from exc

        tokenizer_raw = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer_raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer_raw.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<unk>", "<s>", "</s>"],
        )
        tokenizer_raw.train([str(train_txt_path)], trainer)
        tokenizer_raw.save(str(LOGS_DIR / "bpe_tokenizer.json"))
        tokenizer = _HFBPEWrapper(tokenizer_raw)
    else:
        raise ValueError(f"Unknown tokenizer backend: {backend}")

    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    if bos_id is not None:
        train_ids = [bos_id] + train_ids
        val_ids = [bos_id] + val_ids
    if eos_id is not None:
        train_ids = train_ids + [eos_id]
        val_ids = val_ids + [eos_id]

    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long)

    TOKENIZER_META_PATH.write_text(
        json.dumps(
            {
                "backend": backend,
                "vocab_size": vocab_size,
                "tokenizer_file": "bpe_tokenizer.json" if backend == "hf_bpe" else "spm.model",
            }
        ),
        encoding="utf-8",
    )

    return tokenizer, train_data, val_data


def main():
    torch.manual_seed(TRAIN_CONFIG["seed"])

    train_text, val_text = load_text_data()
    tokenizer, train_data, val_data = build_tokenizer_and_tensors(
        train_text,
        val_text,
        TOKENIZER_CONFIG["vocab_size"],
    )

    model = Llama(
        vocab_size=TRAIN_CONFIG["vocab_size"],
        context_length=MODEL_CONFIG["context_length"],
        d_model=MODEL_CONFIG["d_model"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        d_ff=MODEL_CONFIG["d_ff"],
        rope_theta=MODEL_CONFIG["rope_theta"],
        weights=None,
    ).to(device=DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()

    best_val_loss = float("inf")
    best_eval_state = None
    best_train_loss = float("inf")
    best_train_state = None
    patience_counter = 0
    loss_log_lines = []
    eval_steps = []
    train_losses = []
    val_losses = []

    for step in range(TRAIN_CONFIG["max_iters"]):
        lr = get_lr(
            step,
            TRAIN_CONFIG["learning_rate"],
            TRAIN_CONFIG["min_lr"],
            TRAIN_CONFIG["warmup_iters"],
            TRAIN_CONFIG["max_iters"],
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % TRAIN_CONFIG["eval_interval"] == 0 or step == TRAIN_CONFIG["max_iters"] - 1:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                TRAIN_CONFIG["eval_iters"],
                TRAIN_CONFIG["batch_size"],
                MODEL_CONFIG["context_length"],
                DEVICE,
            )
            train_loss = losses["train"].item()
            val_loss = losses["val"].item()

            if train_loss < best_train_loss - TRAIN_CONFIG["min_delta"]:
                best_train_loss = train_loss
                best_train_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if val_loss < best_val_loss - TRAIN_CONFIG["min_delta"]:
                best_val_loss = val_loss
                best_eval_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            line = (
                f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, "
                f"best val {best_val_loss:.4f}, lr {lr:.2e}"
            )
            print(line)
            loss_log_lines.append(line)
            eval_steps.append(step)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if TRAIN_CONFIG["use_early_stopping"] and patience_counter >= TRAIN_CONFIG["patience"]:
                stop_msg = (
                    f"early stopping at step {step} "
                    f"(no val improvement for {TRAIN_CONFIG['patience']} evals)"
                )
                print(stop_msg)
                loss_log_lines.append(stop_msg)
                break

        xb, yb = get_batch(
            train_data,
            TRAIN_CONFIG["batch_size"],
            MODEL_CONFIG["context_length"],
            DEVICE,
        )
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
        optimizer.step()

        if step % SAMPLE_CONFIG["interval"] == 0:
            with torch.no_grad():
                sample_context = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)
                sample_ids = model.generate(
                    sample_context,
                    max_new_tokens=SAMPLE_CONFIG["max_new_tokens"],
                    temperature=SAMPLING_CONFIG["temperature"],
                    top_k=SAMPLING_CONFIG["top_k"],
                    eos_id=eos_id,
                )[0].tolist()
                sample_text = tokenizer.decode(sample_ids)
            sample_line = f"sample step {step}: {sample_text}"
            print(sample_line)
            loss_log_lines.append(sample_line)

    if best_train_state is not None:
        torch.save(best_train_state, LOGS_DIR / "mini_llama_best_train.pt")

    if best_eval_state is not None:
        torch.save(best_eval_state, LOGS_DIR / "mini_llama_best_eval.pt")
        # Backward-compatible filename: keep writing eval-best weights here.
        torch.save(best_eval_state, LOGS_DIR / "mini_llama_best.pt")
        model.load_state_dict(best_eval_state)
        model.to(device=DEVICE)

    context = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

    generated_ids = model.generate(
        context,
        max_new_tokens=SAMPLING_CONFIG["max_new_tokens"],
        temperature=SAMPLING_CONFIG["temperature"],
        top_k=SAMPLING_CONFIG["top_k"],
        eos_id=eos_id,
    )[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    print(generated_text)

    (LOGS_DIR / "loss_log.txt").write_text("\n".join(loss_log_lines) + "\n", encoding="utf-8")

    if eval_steps:
        plt.figure(figsize=(9, 5))
        plt.plot(eval_steps, train_losses, label="train loss", linewidth=2)
        plt.plot(eval_steps, val_losses, label="val loss", linewidth=2)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Curves")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(LOGS_DIR / "loss_curves.png", dpi=150)
        plt.close()

    (LOGS_DIR / "generated.txt").write_text(generated_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
