import sys
import time
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from test_dataset import SkyScriptDataset
from mim import MIMModel, mim_loss
from vit import SmallViT


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset" / "manifests" / "train_200k.csv"


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Batch size from command line
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    print("Device:", device)
    print("Batch size:", batch_size)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = SkyScriptDataset(
        MANIFEST,
        image_size=224,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    vit = SmallViT(
        img_size=224,
        patch_size=16,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_dim=2048,
    )

    model = MIMModel(vit).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05,
    )

    # --------------------------------------------------
    # Get real batch
    # --------------------------------------------------

    batch = next(iter(loader))

    images = batch["image"].to(
        device,
        non_blocking=True,
    )

    print("Batch shape:", images.shape)

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    model.train()

    predictions, targets, mask = model(
        images,
        mask_ratio=0.5,
    )

    loss = mim_loss(
        predictions,
        targets,
        mask,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # --------------------------------------------------
    # Benchmark
    # --------------------------------------------------

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    predictions, targets, mask = model(
        images,
        mask_ratio=0.5,
    )

    loss = mim_loss(
        predictions,
        targets,
        mask,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    actual_batch_size = images.size(0)

    samples_per_second = (
        actual_batch_size / elapsed
    )

    print()
    print("Benchmark")
    print("---------")

    print(
        "Time per batch:",
        round(elapsed, 4),
        "seconds",
    )

    print(
        "Samples / second:",
        round(samples_per_second, 2),
    )

    print(
        "MIM loss:",
        round(loss.item(), 6),
    )

    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            "Peak GPU memory:",
            round(peak_memory, 3),
            "GB",
        )

    # --------------------------------------------------
    # Rough epoch estimate
    # --------------------------------------------------

    epoch_seconds = (
        len(dataset) / samples_per_second
    )

    print()
    print("Rough 1-epoch estimate:")

    print(
        round(epoch_seconds / 60, 2),
        "minutes",
    )


if __name__ == "__main__":
    main()