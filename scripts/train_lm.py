"""
SkyVLM language-model pretraining on FineWeb-Edu + SkyScript captions.

    python scripts/train_lm.py                  # fresh run
    python scripts/train_lm.py --resume         # continue from checkpoints/lm/lm_latest.pt

Keep --max-steps / --batch-size / --grad-accum / --warmup-steps identical
when resuming (the LR schedule and data order depend on them).
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from lm import SkyVLMForCausalLM, count_parameters


# ============================================================
# Paths
# ============================================================

FINEWEB_PATH = ROOT / "dataset" / "text" / "fineweb_edu" / "fineweb_edu_1m.txt"
SKYSCRIPT_PATH = ROOT / "dataset" / "manifests" / "train_200k.csv"
TOKENIZER_PATH = ROOT / "dataset" / "tokenizer" / "skyvlm.model"
TOKEN_DIR = ROOT / "dataset" / "tokens"
CHECKPOINT_DIR = ROOT / "checkpoints" / "lm"


# ============================================================
# Configuration
# ============================================================

SEED = 42

VOCAB_SIZE = 16384
SEQ_LEN = 512

MODEL_CFG = dict(
    vocab_size=VOCAB_SIZE,
    dim=512,
    num_layers=20,
    num_heads=8,
    ffn_hidden_dim=1408,
    max_seq_len=SEQ_LEN,
)

# 4 * 16 * 512 = 32,768 tokens/update; 30k updates ~ 1B tokens.
# Check tok/s in the first log lines and set MAX_TRAIN_STEPS to fit your time.
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16
MAX_TRAIN_STEPS = 30_000
WARMUP_STEPS = 1000

LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

LOG_EVERY = 20
EVAL_EVERY = 500
SAVE_EVERY = 500        # overwrites lm_latest.pt (with optimizer, for resume)
KEEP_EVERY = 5000       # permanent model-only snapshot

FINEWEB_MAX_DOCS = 1_000_000

# Every VAL_EVERY-th document goes to validation (~1%), per corpus.
VAL_EVERY = 100
EVAL_SEQS = 1024        # sequences per corpus per evaluation

# Share of training *sequences* drawn from SkyScript captions.
SKY_FRACTION = 0.10

NUM_WORKERS = 0         # slicing a memmap is cheap; workers would copy it

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Tokenization (cached as uint16 .npy)
# ============================================================

def read_fineweb():
    with open(FINEWEB_PATH, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= FINEWEB_MAX_DOCS:
                break
            text = line.strip()
            if text:
                yield text


def read_captions():
    with open(SKYSCRIPT_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row["caption"].strip()
            if text:
                yield text


def encode_stream(texts, tokenizer, name, batch_docs=5000):
    """
    Tokenize an iterable of documents, appending EOS after each.
    Returns (train_tokens, val_tokens) as uint16 arrays. The train/val split
    is by document, so no document straddles the boundary.
    """

    eos = tokenizer.eos_id()
    train, val = [], []
    batch, flags = [], []
    n_docs = 0
    n_tokens = 0
    start = time.time()

    def flush():
        nonlocal n_tokens

        for ids, is_val in zip(tokenizer.encode(batch), flags):
            if not ids:
                continue

            arr = np.array(ids + [eos], dtype=np.uint16)
            n_tokens += len(arr)
            (val if is_val else train).append(arr)

        batch.clear()
        flags.clear()

    for text in texts:
        batch.append(text)
        flags.append(n_docs % VAL_EVERY == VAL_EVERY - 1)
        n_docs += 1

        if len(batch) >= batch_docs:
            flush()
            print(
                f"\r  {name}: {n_docs:,} docs | {n_tokens:,} tokens | "
                f"{n_docs / (time.time() - start):,.0f} docs/s",
                end="",
                flush=True,
            )

    flush()
    print(f"\r  {name}: {n_docs:,} docs | {n_tokens:,} tokens" + " " * 20)

    return np.concatenate(train), np.concatenate(val)


def load_or_build_tokens(tokenizer):
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    tok_stat = TOKENIZER_PATH.stat()
    key = {
        "tokenizer_size": tok_stat.st_size,
        "tokenizer_mtime_ns": tok_stat.st_mtime_ns,
        "fineweb_docs": FINEWEB_MAX_DOCS,
        "val_every": VAL_EVERY,
    }

    names = ["fineweb_train", "fineweb_val", "sky_train", "sky_val"]
    meta = TOKEN_DIR / "meta.json"

    cached = (
        meta.exists()
        and json.loads(meta.read_text()) == key
        and all((TOKEN_DIR / f"{n}.npy").exists() for n in names)
    )

    if cached:
        print("Using cached token arrays")
    else:
        print("Tokenizing corpora (cached to dataset/tokens/)")

        if meta.exists():
            meta.unlink()

        fw_train, fw_val = encode_stream(read_fineweb(), tokenizer, "FineWeb")
        sky_train, sky_val = encode_stream(read_captions(), tokenizer, "SkyScript")

        for n, arr in zip(names, [fw_train, fw_val, sky_train, sky_val]):
            np.save(TOKEN_DIR / f"{n}.npy", arr)

        # Written last: a half-finished build never looks valid.
        meta.write_text(json.dumps(key))

    return {n: np.load(TOKEN_DIR / f"{n}.npy", mmap_mode="r") for n in names}


# ============================================================
# Datasets
# ============================================================

def num_sequences(tokens):
    return max(0, (len(tokens) - 1) // SEQ_LEN)


class MixedTokens(Dataset):
    """
    Sequence-level mix of two token arrays. Each sample is one contiguous
    SEQ_LEN+1 window from a single corpus; SkyScript is drawn with
    probability sky_frac. Each corpus is consumed in shuffled passes, so the
    order is fixed by `seed` and a resume can skip `skip` samples exactly.
    """

    def __init__(self, fineweb, sky, num_samples, sky_frac, seed, skip=0):
        self.arrays = (fineweb, sky)

        rng = np.random.default_rng(seed)
        source = (rng.random(num_samples) < sky_frac).astype(np.int8)
        index = np.zeros(num_samples, dtype=np.int64)

        self.epochs = []

        for s, tokens in enumerate(self.arrays):
            mask = source == s
            count = int(mask.sum())
            n_seq = num_sequences(tokens)

            if count == 0:
                self.epochs.append(0.0)
                continue

            assert n_seq > 0, f"corpus {s} has no full sequences"

            passes = -(-count // n_seq)
            order = np.concatenate([rng.permutation(n_seq) for _ in range(passes)])
            index[mask] = order[:count]
            self.epochs.append(count / n_seq)

        self.source = source[skip:]
        self.index = index[skip:]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        start = int(self.index[i]) * SEQ_LEN
        tokens = self.arrays[int(self.source[i])]
        chunk = np.asarray(tokens[start:start + SEQ_LEN + 1], dtype=np.int64)
        t = torch.from_numpy(chunk)

        return t[:-1], t[1:]


class SequentialTokens(Dataset):
    """First `limit` windows of one array, in order (validation)."""

    def __init__(self, tokens, limit=EVAL_SEQS):
        self.tokens = tokens
        self.n = min(num_sequences(tokens), limit)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * SEQ_LEN
        chunk = np.asarray(self.tokens[start:start + SEQ_LEN + 1], dtype=np.int64)
        t = torch.from_numpy(chunk)

        return t[:-1], t[1:]


# ============================================================
# Optimizer / schedule
# ============================================================

def build_optimizer(model):
    # No weight decay on 1-D params (RMSNorm gains).
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        fused=DEVICE.type == "cuda",
    )


def get_lr(step, max_steps, warmup_steps):
    if step < warmup_steps:
        return LEARNING_RATE * (step + 1) / warmup_steps

    if step >= max_steps:
        return MIN_LR

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return MIN_LR + cosine * (LEARNING_RATE - MIN_LR)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loaders):
    """Token-weighted validation loss per corpus, plus the mix-weighted total."""

    model.eval()
    losses = {}

    for name, loader in loaders.items():
        total, count = 0.0, 0

        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=DEVICE.type == "cuda",
            ):
                _, loss = model(x, y)

            total += loss.item() * y.numel()
            count += y.numel()

        losses[name] = total / count if count else float("inf")

    model.train()

    losses["mix"] = (
        (1 - SKY_FRACTION) * losses["fineweb"] + SKY_FRACTION * losses["sky"]
    )

    return losses


# ============================================================
# Checkpoints
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    tokens_seen,
    train_loss,
    val_loss,
    best_val,
):
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": MODEL_CFG,
        "tokens_seen": tokens_seen,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val,
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=MAX_TRAIN_STEPS)
    p.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--save-every", type=int, default=SAVE_EVERY)
    p.add_argument("--keep-every", type=int, default=KEEP_EVERY)

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(TOKENIZER_PATH))
    assert tokenizer.vocab_size() == VOCAB_SIZE

    tokens = load_or_build_tokens(tokenizer)

    tok_per_update = args.batch_size * args.grad_accum * SEQ_LEN

    print("=" * 70)
    print(f"Device            : {DEVICE}")
    print(f"Tokens/update     : {tok_per_update:,}")
    print(f"Max steps         : {args.max_steps:,} "
          f"(~{args.max_steps * tok_per_update / 1e9:.2f}B tokens)")
    print(f"FineWeb train seq : {num_sequences(tokens['fineweb_train']):,}")
    print(f"Sky train seq     : {num_sequences(tokens['sky_train']):,}")
    print("=" * 70)

    # ---------------- model / optimizer ----------------

    model = SkyVLMForCausalLM(**MODEL_CFG).to(DEVICE)
    total_params, _ = count_parameters(model)
    print(f"Parameters: {total_params / 1e6:.2f}M")

    optimizer = build_optimizer(model)

    step = 0
    tokens_seen = 0
    best_val = float("inf")
    latest_path = CHECKPOINT_DIR / "lm_latest.pt"

    if args.resume:
        ckpt = torch.load(latest_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        tokens_seen = ckpt["tokens_seen"]
        best_val = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {step:,}")

    # ---------------- data ----------------

    samples_per_step = args.batch_size * args.grad_accum

    train_ds = MixedTokens(
        tokens["fineweb_train"],
        tokens["sky_train"],
        num_samples=args.max_steps * samples_per_step,
        sky_frac=SKY_FRACTION,
        seed=SEED,
        skip=step * samples_per_step,
    )

    print(
        f"Passes over data  : FineWeb {train_ds.epochs[0]:.2f} | "
        f"SkyScript {train_ds.epochs[1]:.2f}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        drop_last=True,
    )

    val_loaders = {
        name: DataLoader(
            SequentialTokens(tokens[f"{name}_val"]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        for name in ("fineweb", "sky")
    }

    # ---------------- train ----------------

    print("=" * 70)
    print("STARTING PRETRAINING")
    print("=" * 70)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    run_loss = torch.zeros((), device=DEVICE)
    micro = 0
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    start_time = time.time()
    start_tokens = tokens_seen

    for x, y in train_loader:
        lr = get_lr(step, args.max_steps, args.warmup_steps)

        for group in optimizer.param_groups:
            group["lr"] = lr

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=DEVICE.type == "cuda",
        ):
            _, loss = model(x, y)

        (loss / args.grad_accum).backward()

        run_loss += loss.detach()
        micro += 1
        tokens_seen += x.numel()

        if micro % args.grad_accum:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % LOG_EVERY == 0:
            last_train_loss = run_loss.item() / (LOG_EVERY * args.grad_accum)
            run_loss.zero_()

            elapsed = time.time() - start_time
            tok_s = (tokens_seen - start_tokens) / elapsed
            eta_h = (args.max_steps - step) * tok_per_update / tok_s / 3600

            print(
                f"Step {step:6d} | Loss {last_train_loss:.4f} | LR {lr:.2e} | "
                f"Tokens {tokens_seen / 1e6:.1f}M | Tok/s {tok_s:,.0f} | "
                f"ETA {eta_h:.1f}h"
            )

        if step % args.eval_every == 0 or step == args.max_steps:
            val = evaluate(model, val_loaders)
            last_val_loss = val["mix"]

            print(
                f"  val | mix {val['mix']:.4f} | fineweb {val['fineweb']:.4f} "
                f"(ppl {math.exp(val['fineweb']):.1f}) | sky {val['sky']:.4f}"
            )

            if val["mix"] < best_val:
                best_val = val["mix"]
                save_checkpoint(
                    CHECKPOINT_DIR / "lm_best.pt",
                    model, None, step, tokens_seen,
                    last_train_loss, last_val_loss, best_val,
                )
                print("  new best -> lm_best.pt")

        if step % args.save_every == 0:
            save_checkpoint(
                latest_path,
                model, optimizer, step, tokens_seen,
                last_train_loss, last_val_loss, best_val,
            )

        if step % args.keep_every == 0:
            save_checkpoint(
                CHECKPOINT_DIR / f"lm_step_{step:06d}.pt",
                model, None, step, tokens_seen,
                last_train_loss, last_val_loss, best_val,
            )

    # ---------------- final ----------------

    save_checkpoint(
        latest_path,
        model, optimizer, step, tokens_seen,
        last_train_loss, last_val_loss, best_val,
    )
    save_checkpoint(
        CHECKPOINT_DIR / "lm_final.pt",
        model, None, step, tokens_seen,
        last_train_loss, last_val_loss, best_val,
    )

    print("=" * 70)
    print("PRETRAINING COMPLETE")
    print(f"Steps: {step:,} | Tokens: {tokens_seen / 1e9:.4f}B | "
          f"Time: {(time.time() - start_time) / 3600:.2f}h")

    if DEVICE.type == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    print("=" * 70)


if __name__ == "__main__":
    main()