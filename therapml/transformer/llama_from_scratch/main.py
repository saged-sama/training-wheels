from model import inference_test
import os
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from regularization import LabelSmoothing
from model import make_model
from training import run_epoch
from batch_masking import Batch
from loss import SimpleLossCompute
from encoder_decoder import subsequent_mask
from tokenizer import (
    CharTokenizer,
    build_lm_batches,
    build_lm_batches_from_ids,
    load_or_build_token_cache,
    load_tiny_shakespeare_splits,
    load_tiny_stories_splits,
)

def run_tests():
    for _ in range(10):
        inference_test()

RUN_EXAMPLES = True

def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def greedy_generate_from_prompt(model, prompt_ids, pad_idx, max_new_tokens, device):
    generated = prompt_ids[:]
    for _ in range(max_new_tokens):
        src = torch.tensor([generated], dtype=torch.long, device=device)
        seq_mask = (src != pad_idx).unsqueeze(-2)
        causal_mask = subsequent_mask(src.size(1)).type_as(seq_mask.data)
        src_mask = seq_mask & causal_mask
        out = model.forward(src, src, src_mask, src_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        generated.append(int(next_word.item()))
    return generated


def _save_checkpoint(model, tokenizer, path, epoch, epochs, val_loss, model_config, training_config):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "stoi": tokenizer.stoi,
        "itos": tokenizer.itos,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_config": model_config,
        "training_config": training_config,
    }
    torch.save(checkpoint, path)


def example_simple_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    archive_dir = base_dir / "archive"
    checkpoint_path = base_dir / "tiny_shakespeare_transformer.pt"
    cache_path = base_dir / "tiny_shakespeare_tokens.cache.pt"

    splits = load_tiny_shakespeare_splits(archive_dir)
    tokenizer, cached_ids = load_or_build_token_cache(splits, cache_path)
    vocab_size = tokenizer.vocab_size
    N = 2

    criterion = LabelSmoothing(
        size=vocab_size, padding_idx=tokenizer.pad_idx, smoothing=0.0
    )
    model = make_model(vocab_size, vocab_size, N=N).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    quick_mode = os.getenv("LLAMA_QUICK", "0") == "1"
    batch_size = 32
    if quick_mode:
        seq_len = 64
        epochs = 5
        train_max_batches = 300
        val_max_batches = 80
    else:
        seq_len = 128
        epochs = 20
        train_max_batches = 10_000
        val_max_batches = None

    model_config = {"src_vocab": vocab_size, "tgt_vocab": vocab_size, "N": N}
    training_config = {"batch_size": batch_size, "seq_len": seq_len, "epochs": epochs}
    best_path = checkpoint_path.with_stem(checkpoint_path.stem + "_best")
    best_val_loss = float("inf")
    epoch = -1
    try:
        for epoch in range(epochs):
            model.train()
            train_loss, _ = run_epoch(
                build_lm_batches_from_ids(
                    cached_ids["train"],
                    tokenizer,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    shuffle=True,
                    stride=seq_len,
                    max_batches=train_max_batches,
                ),
                model,
                SimpleLossCompute(model.generator, criterion),
                optimizer,
                lr_scheduler,
                mode="train",
            )

            model.eval()
            with torch.no_grad():
                val_loss, _ = run_epoch(
                    build_lm_batches_from_ids(
                        cached_ids["validation"],
                        tokenizer,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        device=device,
                        shuffle=False,
                        stride=seq_len,
                        max_batches=val_max_batches,
                    ),
                    model,
                    SimpleLossCompute(model.generator, criterion),
                    optimizer,
                    lr_scheduler,
                    mode="eval",
                )

            val_loss_f = float(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} | train: {float(train_loss):.4f} | val: {val_loss_f:.4f}"
            )

            # Save after every epoch (overwrites) so the CLI can test anytime
            _save_checkpoint(
                model, tokenizer, checkpoint_path,
                epoch + 1, epochs, val_loss_f, model_config, training_config,
            )
            print(f"  Checkpoint saved (epoch {epoch + 1}): {checkpoint_path}")

            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
                _save_checkpoint(
                    model, tokenizer, best_path,
                    epoch + 1, epochs, val_loss_f, model_config, training_config,
                )
                print(f"  New best ({best_val_loss:.4f}) → {best_path}")

    except KeyboardInterrupt:
        completed = epoch + 1
        print(f"\nInterrupted after epoch {completed}. Saving checkpoint...")
        interrupted_path = checkpoint_path.with_stem(checkpoint_path.stem + "_interrupted")
        _save_checkpoint(
            model, tokenizer, interrupted_path,
            completed, epochs, None, model_config, training_config,
        )
        print(f"Saved to: {interrupted_path}")
        return

    model.eval()
    prompt = "To be, or not to be"
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    decoded = greedy_generate_from_prompt(
        model,
        prompt_ids=prompt_ids,
        pad_idx=tokenizer.pad_idx,
        max_new_tokens=64,
        device=device,
    )
    print(tokenizer.decode(decoded, skip_special_tokens=True))


def example_tiny_stories_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parent
    checkpoint_path = base_dir / "tiny_stories_transformer.pt"
    cache_path = base_dir / "tiny_stories_tokens.cache.pt"

    splits = load_tiny_stories_splits()
    tokenizer, cached_ids = load_or_build_token_cache(splits, cache_path)
    vocab_size = tokenizer.vocab_size
    N = 6

    criterion = LabelSmoothing(
        size=vocab_size, padding_idx=tokenizer.pad_idx, smoothing=0.0
    )
    model = make_model(vocab_size, vocab_size, N=N).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    quick_mode = os.getenv("LLAMA_QUICK", "0") == "1"
    batch_size = 32
    if quick_mode:
        seq_len = 64
        epochs = 5
        train_max_batches = 300
        val_max_batches = 80
    else:
        seq_len = 128
        epochs = 20
        train_max_batches = 10_000
        val_max_batches = None

    model_config = {"src_vocab": vocab_size, "tgt_vocab": vocab_size, "N": N}
    training_config = {"batch_size": batch_size, "seq_len": seq_len, "epochs": epochs}
    best_path = checkpoint_path.with_stem(checkpoint_path.stem + "_best")
    best_val_loss = float("inf")
    epoch = -1
    try:
        for epoch in range(epochs):
            model.train()
            train_loss, _ = run_epoch(
                build_lm_batches_from_ids(
                    cached_ids["train"],
                    tokenizer,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=device,
                    shuffle=True,
                    stride=seq_len,
                    max_batches=train_max_batches,
                ),
                model,
                SimpleLossCompute(model.generator, criterion),
                optimizer,
                lr_scheduler,
                mode="train",
            )

            model.eval()
            with torch.no_grad():
                val_loss, _ = run_epoch(
                    build_lm_batches_from_ids(
                        cached_ids["validation"],
                        tokenizer,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        device=device,
                        shuffle=False,
                        stride=seq_len,
                        max_batches=val_max_batches,
                    ),
                    model,
                    SimpleLossCompute(model.generator, criterion),
                    optimizer,
                    lr_scheduler,
                    mode="eval",
                )

            val_loss_f = float(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs} | train: {float(train_loss):.4f} | val: {val_loss_f:.4f}"
            )

            # Save after every epoch (overwrites) so the CLI can test anytime
            _save_checkpoint(
                model, tokenizer, checkpoint_path,
                epoch + 1, epochs, val_loss_f, model_config, training_config,
            )
            print(f"  Checkpoint saved (epoch {epoch + 1}): {checkpoint_path}")

            if val_loss_f < best_val_loss:
                best_val_loss = val_loss_f
                _save_checkpoint(
                    model, tokenizer, best_path,
                    epoch + 1, epochs, val_loss_f, model_config, training_config,
                )
                print(f"  New best ({best_val_loss:.4f}) → {best_path}")

    except KeyboardInterrupt:
        completed = epoch + 1
        print(f"\nInterrupted after epoch {completed}. Saving checkpoint...")
        interrupted_path = checkpoint_path.with_stem(checkpoint_path.stem + "_interrupted")
        _save_checkpoint(
            model, tokenizer, interrupted_path,
            completed, epochs, None, model_config, training_config,
        )
        print(f"Saved to: {interrupted_path}")
        return

    model.eval()
    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    decoded = greedy_generate_from_prompt(
        model,
        prompt_ids=prompt_ids,
        pad_idx=tokenizer.pad_idx,
        max_new_tokens=64,
        device=device,
    )
    print(tokenizer.decode(decoded, skip_special_tokens=True))


# run_tests()
# show_example(example_simple_model)
show_example(example_tiny_stories_model)