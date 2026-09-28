"""
Supervised reference: fine-tune an encoder end-to-end on the category
labels, as an approximate ceiling for the linear probes.

Same geographic split as the probes: train on split == "train", pick the
best epoch by val Macro-F1, then report val and test (overall and per
region) to checkpoints/eval/probe_results.csv with probe = "finetune".

Examples:
    python finetune_classifier.py checkpoints/mim/main/mim_epoch_100.pt --run-name mim100
    python finetune_classifier.py timm:vit_small_patch16_224.augreg_in21k_ft_in1k --run-name imagenet_vits
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

import linear_probe as lp


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = ROOT / "checkpoints" / "finetune"

IMAGE_SIZE = 224
BATCH_SIZE = 64

EPOCHS = 10
WARMUP_EPOCHS = 1
LEARNING_RATE = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))

SEED = 0

DEVICE = lp.DEVICE


# ============================================================
# Data
# ============================================================

class LabelledImages(Dataset):

    def __init__(self, df, augment):
        self.df = df.reset_index(drop=True)

        if augment:
            # Overhead imagery has no canonical orientation, so vertical
            # flips are as valid as horizontal ones.
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    IMAGE_SIZE,
                    scale=(0.35, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        else:
            # Same preprocessing as the linear probes
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(
            ROOT / "dataset" / "images" / row["image_path"]
        ).convert("RGB")
        return self.transform(image), int(row["label"])


# ============================================================
# Model
# ============================================================

class Classifier(nn.Module):
    """Encoder + linear head on mean-pooled patch tokens."""

    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        tokens = self.encoder(x)
        return self.head(tokens[:, 1:, :].mean(dim=1))


def build_encoder(init):
    if init.startswith(lp.TIMM_PREFIX):
        encoder = lp.TimmEncoder(init[len(lp.TIMM_PREFIX):])
        embed_dim = encoder.model.num_features
    else:
        encoder = lp.load_encoder(None if init == "random" else Path(init))
        embed_dim = encoder.norm.normalized_shape[0]

    # load_encoder freezes weights for probing; unfreeze for fine-tuning
    for param in encoder.parameters():
        param.requires_grad = True

    return encoder.to(DEVICE), embed_dim


# ============================================================
# Training / evaluation
# ============================================================

def get_lr(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return LEARNING_RATE * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    predictions = []

    for images, _ in loader:
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda",
        ):
            logits = model(images.to(DEVICE, non_blocking=True))
        predictions.append(logits.argmax(dim=1).cpu())

    model.train()
    return torch.cat(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("init", help='MIM checkpoint path, "timm:<name>" or "random"')
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split-file", default=str(lp.SPLIT_FILE))
    args = parser.parse_args()

    torch.manual_seed(SEED)

    out_dir = OUTPUT_ROOT / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"

    df, classes = lp.load_split(args.split_file)

    loaders = {
        split: DataLoader(
            LabelledImages(df[df["split"] == split], augment=split == "train"),
            batch_size=BATCH_SIZE,
            shuffle=split == "train",
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            drop_last=split == "train",
        )
        for split in ("train", "val", "test")
    }

    y_val = df.loc[df["split"] == "val", "label"].values

    encoder, embed_dim = build_encoder(args.init)
    model = Classifier(encoder, embed_dim, len(classes)).to(DEVICE)

    print("=" * 70)
    print("Supervised fine-tuning reference")
    print("=" * 70)
    print("Init:", args.init)
    print("Split file:", args.split_file)
    print(f"Train / val / test: {len(loaders['train'].dataset):,} / "
          f"{len(loaders['val'].dataset):,} / {len(loaders['test'].dataset):,}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Epochs {EPOCHS}, batch {BATCH_SIZE}, LR {LEARNING_RATE:g} (cosine), "
          f"label smoothing {LABEL_SMOOTHING}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    steps_per_epoch = len(loaders["train"])
    total_steps = EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    best_f1 = -1.0
    step = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        running_loss = 0.0

        for images, labels in loaders["train"]:
            lr = get_lr(step, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=DEVICE.type == "cuda",
            ):
                loss = criterion(
                    model(images.to(DEVICE, non_blocking=True)),
                    labels.to(DEVICE, non_blocking=True),
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            step += 1

        val_pred = predict(model, loaders["val"])
        val_f1 = f1_score(y_val, val_pred, labels=range(len(classes)),
                          average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_pred)

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "init": args.init, "val_macro_f1": val_f1},
                best_path,
            )

        print(
            f"Epoch {epoch}/{EPOCHS} | train loss {running_loss / steps_per_epoch:.4f} | "
            f"val Macro-F1 {val_f1:.4f} | val acc {val_acc:.4f} | "
            f"{(time.time() - start) / 60:.1f} min{' | best' if improved else ''}",
            flush=True,
        )

    # --------------------------------------------------------
    # Final report with the best-val epoch
    # --------------------------------------------------------

    state = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"\nBest epoch by val Macro-F1: {state['epoch']}")

    for split in ("val", "test"):
        print(f"\n===== {split} =====")
        lp.report_predictions(
            source=f"finetune:{args.init}",
            df=df,
            classes=classes,
            split_file=args.split_file,
            eval_split=split,
            predictions=predict(model, loaders[split]),
            pooling="mean",
            method="finetune",
            details={"converged": f"best epoch {state['epoch']}/{EPOCHS}"},
        )


if __name__ == "__main__":
    main()
