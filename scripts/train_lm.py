import os
import sys
import csv
import math
import time
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader
import sentencepiece as spm


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

FINEWEB_PATH = ROOT / "dataset" / "text" / "fineweb_edu" / "fineweb_edu_1m.txt"
SKYSCRIPT_PATH = ROOT / "dataset" / "manifests" / "train_200k.csv"

TOKENIZER_PATH = ROOT / "dataset" / "tokenizer" / "skyvlm.model"

CHECKPOINT_DIR = ROOT / "checkpoints" / "lm"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Import LM
# ============================================================

sys.path.insert(
    0,
    str(ROOT / "scripts")
)

from lm import SkyVLMForCausalLM, count_parameters


# ============================================================
# Configuration
# ============================================================

SEED = 42

VOCAB_SIZE = 16384
SEQ_LEN = 512

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8

# Effective batch:
# 4 * 8 * 512 = 16,384 tokens/update

MAX_TRAIN_STEPS = 100

LEARNING_RATE = 3e-4
MIN_LR = 3e-5

WEIGHT_DECAY = 0.1

WARMUP_STEPS = 500

GRAD_CLIP = 1.0

LOG_EVERY = 20
EVAL_EVERY = 500
SAVE_EVERY = 500

# Number of FineWeb documents to use.
# We have 1M downloaded.
FINEWEB_MAX_DOCS = 1_000_000

# Validation fraction from the final portion
# of the document streams.
VAL_FRACTION = 0.01

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Tokenizer
# ============================================================

print("=" * 70)
print("SkyVLM Language Model Pretraining")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()

print("Configuration")
print("-" * 70)
print(f"Vocabulary        : {VOCAB_SIZE}")
print(f"Sequence length   : {SEQ_LEN}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Grad accumulation : {GRAD_ACCUM_STEPS}")
print(
    f"Effective tokens/update: "
    f"{BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN:,}"
)
print(f"Max steps         : {MAX_TRAIN_STEPS}")
print(f"Learning rate     : {LEARNING_RATE}")
print(f"Warmup steps      : {WARMUP_STEPS}")
print()


# ============================================================
# Load tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = spm.SentencePieceProcessor()

tokenizer.load(
    str(TOKENIZER_PATH)
)

tokenizer_vocab_size = tokenizer.vocab_size()

print(
    f"Tokenizer vocabulary: "
    f"{tokenizer_vocab_size}"
)

assert tokenizer_vocab_size == VOCAB_SIZE


# ============================================================
# Special tokens
# ============================================================

PAD_ID = tokenizer.pad_id()
BOS_ID = tokenizer.bos_id()
EOS_ID = tokenizer.eos_id()
UNK_ID = tokenizer.unk_id()

print(
    f"Special tokens:"
)

print(
    f"  PAD={PAD_ID}, "
    f"BOS={BOS_ID}, "
    f"EOS={EOS_ID}, "
    f"UNK={UNK_ID}"
)

print()


# ============================================================
# Read SkyScript captions
# ============================================================

def load_skyscript_captions():

    captions = []

    print("Loading SkyScript captions...")

    with open(
        SKYSCRIPT_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            caption = row["caption"].strip()

            if caption:
                captions.append(caption)

    print(
        f"SkyScript captions: "
        f"{len(captions):,}"
    )

    return captions


# ============================================================
# Estimate token counts
# ============================================================

def estimate_sky_tokens(captions):

    total = 0

    sample_count = min(
        len(captions),
        10000
    )

    if sample_count == 0:
        return 0

    for caption in captions[:sample_count]:

        total += len(
            tokenizer.encode(
                caption
            )
        )

    average = total / sample_count

    estimate = int(
        average * len(captions)
    )

    return estimate


# ============================================================
# Load FineWeb documents
# ============================================================

def load_fineweb_documents():

    print(
        f"Reading FineWeb documents..."
    )

    documents = []

    with open(
        FINEWEB_PATH,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as f:

        for i, line in enumerate(f):

            if i >= FINEWEB_MAX_DOCS:
                break

            text = line.strip()

            if text:
                documents.append(text)

    print(
        f"FineWeb documents loaded: "
        f"{len(documents):,}"
    )

    return documents


# ============================================================
# Tokenize documents
# ============================================================

def tokenize_documents(
    documents,
    name
):

    print()
    print(
        f"Tokenizing {name}..."
    )

    tokens = []

    start = time.time()

    for i, text in enumerate(documents):

        encoded = tokenizer.encode(
            text,
            out_type=int
        )

        if not encoded:
            continue

        tokens.extend(encoded)

        # EOS separates documents.
        tokens.append(EOS_ID)

        if (i + 1) % 10000 == 0:

            elapsed = time.time() - start

            rate = (
                (i + 1) / elapsed
            )

            print(
                f"\r  "
                f"{i + 1:,} docs | "
                f"{len(tokens):,} tokens | "
                f"{rate:,.0f} docs/s",
                end=""
            )

    print()

    elapsed = time.time() - start

    print(
        f"{name} tokens: "
        f"{len(tokens):,}"
    )

    print(
        f"Time: {elapsed:.1f}s"
    )

    return tokens


# ============================================================
# Build corpus
# ============================================================

sky_captions = load_skyscript_captions()

estimated_sky_tokens = estimate_sky_tokens(
    sky_captions
)

print(
    f"Estimated SkyScript tokens: "
    f"{estimated_sky_tokens:,}"
)

fineweb_documents = load_fineweb_documents()

# IMPORTANT:
# We tokenize FineWeb first.
# This gives us the actual token count.
fineweb_tokens = tokenize_documents(
    fineweb_documents,
    "FineWeb"
)

sky_tokens = tokenize_documents(
    sky_captions,
    "SkyScript"
)

print()
print("=" * 70)
print("CORPUS SUMMARY")
print("=" * 70)

print(
    f"FineWeb tokens   : "
    f"{len(fineweb_tokens):,}"
)

print(
    f"SkyScript tokens : "
    f"{len(sky_tokens):,}"
)

total_tokens = (
    len(fineweb_tokens)
    + len(sky_tokens)
)

print(
    f"Total tokens     : "
    f"{total_tokens:,}"
)

print(
    f"Total tokens (B) : "
    f"{total_tokens / 1e9:.4f}"
)

print(
    f"SkyScript share  : "
    f"{100 * len(sky_tokens) / total_tokens:.2f}%"
)

print("=" * 70)
print()


# ============================================================
# Split into train / validation
# ============================================================

# Keep the split deterministic.
# FineWeb and SkyScript are each split independently.

def split_tokens(tokens):

    n = len(tokens)

    val_tokens = max(
        SEQ_LEN,
        int(n * VAL_FRACTION)
    )

    train_tokens = tokens[:-val_tokens]
    validation_tokens = tokens[-val_tokens:]

    return train_tokens, validation_tokens


fineweb_train, fineweb_val = split_tokens(
    fineweb_tokens
)

sky_train, sky_val = split_tokens(
    sky_tokens
)


# ============================================================
# Mix corpora
# ============================================================

# 90% FineWeb
# 10% SkyScript at the sequence sampling level.

# Instead of physically duplicating data,
# create a mixed token stream.

MIX_FINEWEB_RATIO = 0.90
MIX_SKYSCRIPT_RATIO = 0.10


def mix_token_streams(
    fineweb,
    sky,
    total_length
):

    output = []

    fine_pos = 0
    sky_pos = 0

    fine_target = int(
        total_length * MIX_FINEWEB_RATIO
    )

    sky_target = (
        total_length - fine_target
    )

    # Add FineWeb.
    while (
        fine_pos < len(fineweb)
        and len(output) < fine_target
    ):

        output.append(
            fineweb[fine_pos]
        )

        fine_pos += 1

    # Add SkyScript.
    sky_output = []

    while (
        sky_pos < len(sky)
        and len(sky_output) < sky_target
    ):

        sky_output.append(
            sky[sky_pos]
        )

        sky_pos += 1

    # Interleave approximately according to ratio.
    mixed = []

    fi = 0
    si = 0

    while (
        fi < len(output)
        or si < len(sky_output)
    ):

        if (
            fi < len(output)
            and si < len(sky_output)
        ):

            # Keep approximate 9:1 ratio.
            for _ in range(9):

                if fi >= len(output):
                    break

                mixed.append(
                    output[fi]
                )

                fi += 1

            mixed.append(
                sky_output[si]
            )

            si += 1

        elif fi < len(output):

            mixed.append(
                output[fi]
            )

            fi += 1

        else:

            mixed.append(
                sky_output[si]
            )

            si += 1

    return mixed


train_tokens = mix_token_streams(
    fineweb_train,
    sky_train,
    min(
        len(fineweb_train),
        int(
            len(fineweb_train)
            / MIX_FINEWEB_RATIO
        )
    )
)

val_tokens = mix_token_streams(
    fineweb_val,
    sky_val,
    min(
        len(fineweb_val),
        int(
            len(fineweb_val)
            / MIX_FINEWEB_RATIO
        )
    )
)

print(
    f"Mixed training tokens: "
    f"{len(train_tokens):,}"
)

print(
    f"Mixed validation tokens: "
    f"{len(val_tokens):,}"
)

print()


# ============================================================
# Token Dataset
# ============================================================

class TokenDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        tokens,
        seq_len
    ):

        self.tokens = tokens
        self.seq_len = seq_len

        self.num_samples = (
            (len(tokens) - 1)
            // seq_len
        )

    def __len__(self):

        return self.num_samples

    def __getitem__(self, idx):

        start = idx * self.seq_len

        end = start + self.seq_len + 1

        chunk = self.tokens[
            start:end
        ]

        if len(chunk) < self.seq_len + 1:

            chunk = (
                chunk
                + [PAD_ID]
                * (
                    self.seq_len
                    + 1
                    - len(chunk)
                )
            )

        x = torch.tensor(
            chunk[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            chunk[1:],
            dtype=torch.long
        )

        return x, y


train_dataset = TokenDataset(
    train_tokens,
    SEQ_LEN
)

val_dataset = TokenDataset(
    val_tokens,
    SEQ_LEN
)

print(
    f"Train sequences: "
    f"{len(train_dataset):,}"
)

print(
    f"Validation sequences: "
    f"{len(val_dataset):,}"
)

print()


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# Model
# ============================================================

print("=" * 70)
print("BUILDING MODEL")
print("=" * 70)

model = SkyVLMForCausalLM(
    vocab_size=VOCAB_SIZE,
    dim=512,
    num_layers=20,
    num_heads=8,
    ffn_hidden_dim=1408,
    max_seq_len=SEQ_LEN
).to(DEVICE)

total_params, trainable_params = count_parameters(
    model
)

print(
    f"Parameters: "
    f"{total_params:,}"
)

print(
    f"Parameters: "
    f"{total_params / 1e6:.2f}M"
)

print()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Learning Rate Schedule
# ============================================================

def get_lr(step):

    if step < WARMUP_STEPS:

        return (
            LEARNING_RATE
            * (step + 1)
            / WARMUP_STEPS
        )

    if step >= MAX_TRAIN_STEPS:

        return MIN_LR

    progress = (
        step - WARMUP_STEPS
    ) / (
        MAX_TRAIN_STEPS
        - WARMUP_STEPS
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    return (
        MIN_LR
        + cosine
        * (
            LEARNING_RATE
            - MIN_LR
        )
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for x, y in val_loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=DEVICE.type == "cuda"
        ):

            _, loss = model(
                x,
                y
            )

        total_loss += loss.item()
        total_batches += 1

    model.train()

    if total_batches == 0:
        return float("inf")

    return (
        total_loss
        / total_batches
    )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    step,
    train_loss,
    val_loss,
    tokens_seen
):

    path = (
        CHECKPOINT_DIR
        / f"lm_step_{step:06d}.pt"
    )

    checkpoint = {

        "step": step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "tokens_seen":
            tokens_seen,

        "config": {

            "vocab_size":
                VOCAB_SIZE,

            "dim":
                512,

            "num_layers":
                20,

            "num_heads":
                8,

            "ffn_hidden_dim":
                1408,

            "max_seq_len":
                SEQ_LEN
        }
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved: {path}"
    )


# ============================================================
# Training
# ============================================================

print("=" * 70)
print("STARTING PRETRAINING")
print("=" * 70)

print()

model.train()

step = 0
tokens_seen = 0

running_loss = 0.0
running_steps = 0

best_val_loss = float("inf")

start_time = time.time()

optimizer.zero_grad(
    set_to_none=True
)

while step < MAX_TRAIN_STEPS:

    for x, y in train_loader:

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # ----------------------------------------------------
        # Move to GPU
        # ----------------------------------------------------

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=DEVICE.type == "cuda"
        ):

            _, loss = model(
                x,
                y
            )

            loss_for_backward = (
                loss
                / GRAD_ACCUM_STEPS
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss_for_backward.backward()

        running_loss += loss.item()
        running_steps += 1

        tokens_seen += (
            BATCH_SIZE
            * SEQ_LEN
        )

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        if (
            running_steps
            % GRAD_ACCUM_STEPS
            == 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            step += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                step % LOG_EVERY == 0
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                avg_loss = (
                    running_loss
                    / (LOG_EVERY * GRAD_ACCUM_STEPS)
                )

                tokens_per_sec = (
                    tokens_seen
                    / elapsed
                )

                print(
                    f"Step {step:6d} | "
                    f"Loss {avg_loss:.4f} | "
                    f"LR {lr:.2e} | "
                    f"Tokens {tokens_seen / 1e6:.2f}M | "
                    f"Tok/s {tokens_per_sec:,.0f}"
                )

                running_loss = 0.0

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if (
                step % EVAL_EVERY == 0
            ):

                val_loss = evaluate()

                print(
                    f"Validation loss: "
                    f"{val_loss:.4f}"
                )

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    best_path = (
                        CHECKPOINT_DIR
                        / "lm_best.pt"
                    )

                    torch.save(
                        {
                            "step": step,

                            "model_state_dict":
                                model.state_dict(),

                            "optimizer_state_dict":
                                optimizer.state_dict(),

                            "train_loss":
                                avg_loss,

                            "val_loss":
                                val_loss,

                            "tokens_seen":
                                tokens_seen
                        },
                        best_path
                    )

                    print(
                        f"New best checkpoint: "
                        f"{best_path}"
                    )

            # ------------------------------------------------
            # Periodic checkpoint
            # ------------------------------------------------

            if (
                step % SAVE_EVERY == 0
            ):

                save_checkpoint(
                    step,
                    avg_loss
                    if 'avg_loss' in locals()
                    else loss.item(),
                    val_loss
                    if 'val_loss' in locals()
                    else float("nan"),
                    tokens_seen
                )

            # ------------------------------------------------
            # Stop
            # ------------------------------------------------

            if step >= MAX_TRAIN_STEPS:

                break

    if step >= MAX_TRAIN_STEPS:

        break


# ============================================================
# Final save
# ============================================================

final_path = (
    CHECKPOINT_DIR
    / "lm_final.pt"
)

torch.save(
    {
        "step": step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "tokens_seen":
            tokens_seen
    },
    final_path
)

elapsed = (
    time.time()
    - start_time
)

print()
print("=" * 70)
print("PRETRAINING COMPLETE")
print("=" * 70)

print(
    f"Steps: {step:,}"
)

print(
    f"Tokens seen: "
    f"{tokens_seen:,}"
)

print(
    f"Tokens seen: "
    f"{tokens_seen / 1e9:.4f}B"
)

print(
    f"Training time: "
    f"{elapsed / 3600:.2f} hours"
)

if DEVICE.type == "cuda":

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"Peak VRAM: "
        f"{peak:.3f} GB"
    )

print(
    f"Final checkpoint: "
    f"{final_path}"
)

print("=" * 70)