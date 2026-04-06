import csv
import json
import os
import sys
from pathlib import Path

import torch

from batch_masking import Batch
from encoder_decoder import subsequent_mask


class LanguageModelBatch:
    def __init__(self, tokens, pad=0):
        self.src = tokens[:, :-1]
        self.tgt = tokens[:, :-1]
        self.tgt_y = tokens[:, 1:]
        self.src_mask = self.make_src_mask(self.src, pad)
        self.tgt_mask = Batch.make_std_mask(self.tgt, pad)
        self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_src_mask(src, pad):
        src_pad_mask = (src != pad).unsqueeze(-2)
        src_causal_mask = subsequent_mask(src.size(-1)).type_as(src_pad_mask.data)
        return src_pad_mask & src_causal_mask


class CharTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    @classmethod
    def from_text(cls, text):
        special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        unique_chars = sorted(set(text))
        vocab = special_tokens + unique_chars
        stoi = {token: idx for idx, token in enumerate(vocab)}
        itos = {idx: token for token, idx in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def pad_idx(self):
        return self.stoi[self.pad_token]

    @property
    def bos_idx(self):
        return self.stoi[self.bos_token]

    @property
    def eos_idx(self):
        return self.stoi[self.eos_token]

    @property
    def unk_idx(self):
        return self.stoi[self.unk_token]

    def encode(self, text, add_bos=False, add_eos=False):
        token_ids = [self.stoi.get(ch, self.unk_idx) for ch in text]
        if add_bos:
            token_ids = [self.bos_idx] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_idx]
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        special = {self.pad_token, self.bos_token, self.eos_token, self.unk_token}
        chars = []
        for idx in token_ids:
            token = self.itos.get(int(idx), self.unk_token)
            if skip_special_tokens and token in special:
                continue
            chars.append(token)
        return "".join(chars)


def _load_text_column(csv_path, column_name="text"):
    csv.field_size_limit(sys.maxsize)
    parts = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column_name not in reader.fieldnames:
            raise ValueError(f"Expected '{column_name}' column in {csv_path}")
        for row in reader:
            parts.append(row.get(column_name, ""))
    return "\n".join(parts)


def load_tiny_stories_splits():
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories")
    train_text = "\n".join(dataset["train"]["text"])
    val_text = "\n".join(dataset["validation"]["text"])
    return {"train": train_text, "validation": val_text}


def load_tiny_shakespeare_splits(archive_dir):
    archive_path = Path(archive_dir)
    split_files = {
        "train": archive_path / "train.csv",
        "validation": archive_path / "validation.csv",
        "test": archive_path / "test.csv",
    }
    return {
        split_name: _load_text_column(path, column_name="text")
        for split_name, path in split_files.items()
    }


def build_lm_batches(
    text,
    tokenizer,
    batch_size,
    seq_len,
    device=None,
    shuffle=False,
    stride=None,
    max_batches=None,
):
    window_size = seq_len + 1
    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(token_ids) <= window_size:
        return

    if stride is None:
        stride = seq_len
    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    windows = []
    for start in range(0, len(token_ids) - window_size, stride):
        window = token_ids[start : start + window_size]
        windows.append(window)

    if shuffle:
        indices = torch.randperm(len(windows)).tolist()
    else:
        indices = list(range(len(windows)))

    yielded_batches = 0
    for start in range(0, len(indices), batch_size):
        idx_slice = indices[start : start + batch_size]
        if len(idx_slice) < batch_size:
            continue

        token_batch = torch.tensor([windows[i] for i in idx_slice], dtype=torch.long)

        if device is not None:
            token_batch = token_batch.to(device)

        yield LanguageModelBatch(tokens=token_batch, pad=tokenizer.pad_idx)
        yielded_batches += 1
        if max_batches is not None and yielded_batches >= max_batches:
            break


def build_lm_batches_from_ids(
    token_ids,
    tokenizer,
    batch_size,
    seq_len,
    device=None,
    shuffle=False,
    stride=None,
    max_batches=None,
):
    """Like build_lm_batches but accepts pre-tokenized IDs (list or 1-D tensor).

    Use this with token IDs loaded from a cache to skip re-tokenizing each epoch.
    """
    window_size = seq_len + 1
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.as_tensor(token_ids, dtype=torch.int32)
    n = len(token_ids)
    if n <= window_size:
        return
    if stride is None:
        stride = seq_len
    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    start_positions = list(range(0, n - window_size, stride))
    num_windows = len(start_positions)

    if shuffle:
        perm = torch.randperm(num_windows).tolist()
    else:
        perm = list(range(num_windows))

    yielded_batches = 0
    for batch_start in range(0, num_windows, batch_size):
        idx_slice = perm[batch_start : batch_start + batch_size]
        if len(idx_slice) < batch_size:
            continue

        token_batch = torch.stack(
            [
                token_ids[start_positions[i] : start_positions[i] + window_size].long()
                for i in idx_slice
            ]
        )

        if device is not None:
            token_batch = token_batch.to(device)

        yield LanguageModelBatch(tokens=token_batch, pad=tokenizer.pad_idx)
        yielded_batches += 1
        if max_batches is not None and yielded_batches >= max_batches:
            break


def load_or_build_token_cache(splits, cache_path):
    """Load tokenizer and token IDs from a .pt cache, or build and save them.

    Tokenizes all text splits on first call and saves to ``cache_path``.
    Subsequent calls load directly from disk — much faster than re-tokenizing.
    Delete the cache file to force a full rebuild.

    Args:
        splits: dict mapping split name -> raw text string
        cache_path: path to the cache file (will be created on first run)

    Returns:
        tuple: (CharTokenizer, dict[str, torch.Tensor]) with int32 token ID tensors
    """
    cache_path = Path(cache_path)
    json_path = cache_path.with_suffix(".json")

    def _write_token_json_debug(tokenizer, token_ids, output_path):
        include_full = os.getenv("LLAMA_TOKEN_JSON_FULL", "0") == "1"
        max_ids_per_split = int(os.getenv("LLAMA_TOKEN_JSON_MAX_IDS", "2000"))

        splits_payload = {}
        for split_name, ids in token_ids.items():
            ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
            visible_ids = ids_list if include_full else ids_list[:max_ids_per_split]
            splits_payload[split_name] = {
                "num_token_ids": len(ids_list),
                "shown_token_ids": visible_ids,
                "shown_token_ids_count": len(visible_ids),
                "truncated": (not include_full) and (len(ids_list) > len(visible_ids)),
                "decoded_preview": tokenizer.decode(visible_ids, skip_special_tokens=False),
            }

        payload = {
            "format": "therapml_token_cache_debug_v1",
            "cache_file": str(cache_path),
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": {
                "pad": tokenizer.pad_idx,
                "bos": tokenizer.bos_idx,
                "eos": tokenizer.eos_idx,
                "unk": tokenizer.unk_idx,
            },
            "sample_mode": "full" if include_full else "sample",
            "max_ids_per_split": None if include_full else max_ids_per_split,
            "stoi": tokenizer.stoi,
            "splits": splits_payload,
        }

        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"Token debug JSON saved: {output_path}", flush=True)

    if cache_path.exists():
        print(f"Loading token cache: {cache_path}", flush=True)
        # weights_only=False is safe here — we wrote this file ourselves
        cache = torch.load(cache_path, weights_only=False)
        stoi = cache["stoi"]
        itos = {int(k): v for k, v in cache["itos"].items()}
        tokenizer = CharTokenizer(stoi=stoi, itos=itos)
        token_ids = cache["token_ids"]

        write_json_on_load = os.getenv("LLAMA_TOKEN_JSON_ON_LOAD", "1") == "1"
        if write_json_on_load and (not json_path.exists() or os.getenv("LLAMA_FORCE_TOKEN_JSON", "0") == "1"):
            _write_token_json_debug(tokenizer, token_ids, json_path)

        return tokenizer, token_ids

    print("Building token cache (first run — will be faster next time)...", flush=True)
    tokenizer = CharTokenizer.from_text(splits["train"])
    token_ids = {}
    for split_name, text in splits.items():
        print(f"  Encoding '{split_name}' ({len(text):,} chars)...", flush=True)
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        token_ids[split_name] = torch.tensor(ids, dtype=torch.int32)

    cache = {
        "stoi": tokenizer.stoi,
        "itos": {str(k): v for k, v in tokenizer.itos.items()},
        "token_ids": token_ids,
    }
    torch.save(cache, cache_path)
    size_mb = cache_path.stat().st_size // (1024 * 1024)
    print(f"Token cache saved: {cache_path} ({size_mb} MB)", flush=True)

    _write_token_json_debug(tokenizer, token_ids, json_path)

    return tokenizer, token_ids