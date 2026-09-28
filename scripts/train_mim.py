import argparse
import os
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from test_dataset import SkyScriptDataset
from vit import SmallViT
from mim import MIMModel, mim_loss
import linear_probe


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    ROOT
    / "dataset"
    / "manifests"
    / "train_200k.csv"
)

# Fixed geographic split; its "quick_probe" column marks the
# 16k train / 4k val sample used during training
PROBE_SPLIT_FILE = (
    ROOT
    / "dataset"
    / "manifests"
    / "probe_geo_split.csv"
)

CHECKPOINT_ROOT = (
    ROOT
    / "checkpoints"
    / "mim"
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224
PATCH_SIZE = 16

BATCH_SIZE = 32

EMBED_DIM = 512
DEPTH = 8
NUM_HEADS = 8
MLP_DIM = 2048

NORM_PIX_LOSS = True

LEARNING_RATE = 1e-4        # peak, reached after warmup
MIN_LR = 1e-6
WEIGHT_DECAY = 0.05

# Overridable so the pipeline can reduce RAM use
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))

# Fewer workers for the quick probe, which runs while the
# training loader's persistent workers are still alive
PROBE_WORKERS = min(4, NUM_WORKERS)

LOG_EVERY = 100
KEEP_EVERY = 10             # keep a permanent checkpoint every N epochs

PROBE_EVERY = 10

SEED = 42

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mask-ratio",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--run-name",
        default="main",
        help="Checkpoints go to checkpoints/mim/<run-name>/",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from <run-name>/mim_latest.pt",
    )

    return parser.parse_args()


# ============================================================
# Learning-rate schedule: linear warmup, then cosine decay
# ============================================================

def get_lr(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return LEARNING_RATE * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

    return MIN_LR + 0.5 * (LEARNING_RATE - MIN_LR) * (
        1 + math.cos(math.pi * progress)
    )


# ============================================================
# Quick linear probe on the fixed quick-probe sample
# ============================================================

def build_probe_split():
    # Fixed 16k train / 4k val sample from the geographic split
    df, classes = linear_probe.load_split(PROBE_SPLIT_FILE)

    subset = df[df["quick_probe"] != ""].reset_index(drop=True)

    train_idx = torch.tensor(
        (subset["quick_probe"] == "train").values
    ).nonzero().squeeze(1)

    val_idx = torch.tensor(
        (subset["quick_probe"] == "val").values
    ).nonzero().squeeze(1)

    return (
        subset,
        train_idx,
        val_idx,
        len(classes),
    )


def quick_probe(vit, probe_split):
    subset, train_idx, val_idx, num_classes = probe_split

    loader = DataLoader(
        linear_probe.SkyScriptDataset(subset),
        batch_size=128,
        shuffle=False,
        num_workers=PROBE_WORKERS,
        pin_memory=True,
    )

    vit.eval()

    features = {"cls": [], "mean": []}
    labels = []

    for images, y in loader:
        with torch.no_grad(), torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=DEVICE.type == "cuda",
        ):
            output = vit(images.to(DEVICE, non_blocking=True))

        features["cls"].append(output[:, 0, :].float().cpu())
        features["mean"].append(output[:, 1:, :].float().mean(dim=1).cpu())
        labels.append(y)

    vit.train()

    labels = torch.cat(labels)
    results = {}

    for pooling, chunks in features.items():
        pooled = torch.cat(chunks)

        accuracy, macro_f1, _ = linear_probe.train_probe(
            pooled[train_idx],
            labels[train_idx],
            pooled[val_idx],
            labels[val_idx],
            num_classes,
        )

        results[pooling] = (accuracy, macro_f1)

    return results


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    epoch,
    mask_ratio,
    global_step,
    warmup_steps,
    total_steps,
):
    model.train()

    running_loss = 0.0
    num_batches = 0

    use_amp = DEVICE.type == "cuda"
    start = time.time()

    for step, batch in enumerate(loader):

        lr = get_lr(global_step, warmup_steps, total_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            predictions, targets, mask = model(
                images,
                mask_ratio=mask_ratio,
            )

            loss = mim_loss(
                predictions,
                targets,
                mask,
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        global_step += 1

        if step % LOG_EVERY == 0:
            images_per_sec = num_batches * BATCH_SIZE / (time.time() - start)
            print(
                f"Epoch {epoch} | "
                f"Step {step:05d}/{len(loader):05d} | "
                f"Loss {loss.item():.6f} | "
                f"LR {lr:.2e} | "
                f"{images_per_sec:.0f} img/s"
            )

    return running_loss / num_batches, global_step


def append_csv(path, row):
    new_file = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def log_probe(run_dir, epoch, results):
    for pooling, (accuracy, macro_f1) in results.items():
        print(
            f"Probe @ epoch {epoch} | {pooling:4s} | "
            f"val macro-F1 {macro_f1:.4f} | val acc {accuracy:.4f}"
        )
        append_csv(
            run_dir / "probe_log.csv",
            {
                "epoch": epoch,
                "pooling": pooling,
                "val_macro_f1": f"{macro_f1:.4f}",
                "val_accuracy": f"{accuracy:.4f}",
            },
        )


def save_checkpoint(path, **state):
    # Write to a temp file first so an interruption never
    # leaves a corrupt checkpoint behind.
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def main():
    args = parse_args()

    torch.manual_seed(SEED)

    run_dir = CHECKPOINT_ROOT / args.run_name
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    latest_path = run_dir / "mim_latest.pt"

    print("=" * 70)
    print("SkyScript MIM Pretraining")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("Manifest:", MANIFEST)
    print("Run directory:", run_dir)
    print("Batch size:", BATCH_SIZE)
    print("Mask ratio:", args.mask_ratio)
    print("Norm pixel loss:", NORM_PIX_LOSS)
    print("Epochs:", args.epochs)
    print("Warmup epochs:", args.warmup_epochs)
    print(f"LR: {LEARNING_RATE:.1e} -> {MIN_LR:.1e} (cosine)")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    dataset = SkyScriptDataset(
        MANIFEST,
        image_size=IMAGE_SIZE,
        augment=True,
    )

    print(
        "Dataset size:",
        len(dataset),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    print(
        "Batches per epoch:",
        len(loader),
    )

    probe_split = build_probe_split()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    vit = SmallViT(
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
    )

    model = MIMModel(
        vit=vit,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        in_channels=3,
        norm_pix_loss=NORM_PIX_LOSS,
    ).to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "MIM parameters:",
        f"{total_params / 1e6:.2f}M",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=DEVICE.type == "cuda",
    )

    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    config = {
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "batch_size": BATCH_SIZE,
        "mask_ratio": args.mask_ratio,
        "mask_token": "learned",
        "norm_pix_loss": NORM_PIX_LOSS,
        "augment": True,
        "embed_dim": EMBED_DIM,
        "depth": DEPTH,
        "num_heads": NUM_HEADS,
        "mlp_dim": MLP_DIM,
        "learning_rate": LEARNING_RATE,
        "min_lr": MIN_LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
    }

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_epoch = 1
    global_step = 0

    if args.resume:
        state = torch.load(
            latest_path,
            map_location=DEVICE,
            weights_only=False,
        )

        if state["config"] != config:
            raise SystemExit(
                "Checkpoint config differs from current settings:\n"
                f"  checkpoint: {state['config']}\n"
                f"  current:    {config}"
            )

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])

        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]

        print(f"Resumed from epoch {state['epoch']}")

    elif latest_path.exists():
        raise SystemExit(
            f"{latest_path} already exists. "
            "Use --resume, or choose a different --run-name."
        )

    else:
        # Baseline: probe the untrained encoder
        log_probe(run_dir, 0, quick_probe(model.vit, probe_split))

    print()
    print("Starting training...")
    print()

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.time()

        epoch_loss, global_step = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            mask_ratio=args.mask_ratio,
            global_step=global_step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        epoch_minutes = (time.time() - epoch_start) / 60

        print()
        print(
            f"Epoch {epoch} complete | "
            f"Average loss: {epoch_loss:.6f} | "
            f"{epoch_minutes:.1f} min"
        )

        if DEVICE.type == "cuda":
            print(
                f"Peak VRAM: "
                f"{torch.cuda.max_memory_allocated() / 1024**3:.3f} GB"
            )

        append_csv(
            run_dir / "train_log.csv",
            {
                "epoch": epoch,
                "loss": f"{epoch_loss:.6f}",
                "minutes": f"{epoch_minutes:.2f}",
            },
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": epoch_loss,
            "config": config,
        }

        save_checkpoint(latest_path, **state)

        if epoch % KEEP_EVERY == 0 or epoch == args.epochs:
            save_checkpoint(
                run_dir / f"mim_epoch_{epoch:03d}.pt",
                **state,
            )

        print(
            "Checkpoint saved:",
            latest_path,
        )

        # ----------------------------------------------------
        # Quick probe
        # ----------------------------------------------------

        if epoch % PROBE_EVERY == 0 or epoch == args.epochs:
            log_probe(run_dir, epoch, quick_probe(model.vit, probe_split))

        print()


if __name__ == "__main__":
    main()
