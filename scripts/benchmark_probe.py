"""
Linear-probe benchmark on NWPU-RESISC45 (45 aerial scene classes, clean
labels, standard 18.9k / 6.3k / 6.3k train / val / test split from
timm/resisc45 on Hugging Face).

Frozen features (CLS and mean-pooled patch tokens), standardized with
train statistics, then multinomial logistic regression fitted to
convergence with full-batch L-BFGS on the GPU (float32, then float64). The L2 strength is chosen on
val accuracy; test accuracy (the metric usually reported for RESISC45) and
Macro-F1 are recorded in checkpoints/eval/benchmark_results.csv.

Examples:
    python benchmark_probe.py random checkpoints/mim/main/mim_epoch_100.pt
    python benchmark_probe.py timm:vit_small_patch16_224.augreg_in21k_ft_in1k
"""

import argparse
import csv
import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import linear_probe as lp


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "benchmarks" / "resisc45" / "data"
FEATURE_DIR = ROOT / "checkpoints" / "eval" / "resisc45"
RESULTS_FILE = ROOT / "checkpoints" / "eval" / "benchmark_results.csv"

SPLITS = {"train": "train", "val": "validation", "test": "test"}

IMAGE_SIZE = 224
BATCH_SIZE = 64

# Same L2 grid as the SkyScript probe (mean cross-entropy + 0.5 * l2 * |W|^2)
L2_GRID = lp.L2_GRID

# GPU L-BFGS: float32 steps (~1 ms) until float32's precision floor, then
# float64 (~9 ms on this GPU) until max |grad| < GRAD_TOL.
GRAD_TOL = 1e-6
CHUNK_ITER = 100
MAX_ITER_FP32 = 20_000
MAX_ITER_FP64 = 20_000

# Float64 phase also stops if the loss improves by less than STALL_TOL over
# STALL_CHUNKS chunks (near-unregularized fits with no finite optimum)
STALL_CHUNKS = 10
STALL_TOL = 1e-8


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

class ParquetImages(Dataset):

    def __init__(self, df):
        self.images = df["image"].map(lambda x: x["bytes"]).tolist()
        self.labels = df["label"].tolist()

        # Same preprocessing as the SkyScript probes
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.images[idx])).convert("RGB")
        return self.transform(image), self.labels[idx]


def load_splits():
    return {
        name: pd.read_parquet(DATA_DIR / f"{file}-00000-of-00001.parquet")
        for name, file in SPLITS.items()
    }


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

def load_model(source):
    if source.startswith(lp.TIMM_PREFIX):
        return lp.load_timm_encoder(source[len(lp.TIMM_PREFIX):])
    return lp.load_encoder(None if source == "random" else Path(source))


def cache_key(source):
    if source == "random" or source.startswith(lp.TIMM_PREFIX):
        return source
    path = Path(source)
    return f"{path.resolve()}:{path.stat().st_mtime_ns}"


@torch.inference_mode()
def extract(source, splits):
    cache = FEATURE_DIR / lp.cache_path_for(source).name
    key = cache_key(source)

    if cache.exists():
        data = torch.load(cache, weights_only=True)
        if data.get("source") == key:
            print("Loading cached features:", cache)
            return data["features"]

    model = load_model(source)
    features = {}

    for split, df in splits.items():
        loader = DataLoader(
            ParquetImages(df),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=lp.NUM_WORKERS,
        )
        chunks = {"cls": [], "mean": []}
        start = time.perf_counter()

        for images, _ in loader:
            output = model(images.to(lp.DEVICE, non_blocking=True))
            chunks["cls"].append(output[:, 0, :].float().cpu())
            chunks["mean"].append(output[:, 1:, :].float().mean(dim=1).cpu())

        features[split] = {k: torch.cat(v) for k, v in chunks.items()}
        print(f"  {split}: {len(df):,} images in {time.perf_counter() - start:.0f}s")

    del model
    if lp.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "source": key}, cache)

    return features


# ---------------------------------------------------------
# Probe
# ---------------------------------------------------------

def lbfgs_phase(X, y, l2, W, b, stop):
    """Run L-BFGS in CHUNK_ITER-step chunks until stop(history) is true."""

    W = W.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [W, b],
        lr=1,
        max_iter=CHUNK_ITER,
        tolerance_grad=0,
        tolerance_change=0,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = (
            torch.nn.functional.cross_entropy(X @ W.T + b, y)
            + 0.5 * l2 * (W ** 2).sum()
        )
        loss.backward()
        return loss

    history = []
    while True:
        optimizer.step(closure)
        loss = closure().item()
        grad = max(W.grad.abs().max().item(), b.grad.abs().max().item())
        history.append((loss, grad))
        if stop(history):
            return W.detach(), b.detach(), len(history) * CHUNK_ITER, grad


def fit_logreg(X32, X64, y, l2, W, b):
    """
    Multinomial logistic regression (mean cross-entropy + 0.5 * l2 * |W|^2,
    bias unpenalized) with full-batch L-BFGS on the GPU: fast float32
    steps until the gradient stops shrinking (float32's precision floor),
    then float64 until max |grad| < GRAD_TOL. Near-unregularized fits may
    have no finite optimum; those stop once the loss stops improving and
    are reported as not converged.
    """

    def float32_done(h):
        return (
            h[-1][1] < GRAD_TOL
            or (len(h) > 1 and h[-1][1] > 0.9 * h[-2][1])
            or len(h) * CHUNK_ITER >= MAX_ITER_FP32
        )

    W, b, iters32, _ = lbfgs_phase(X32, y, l2, W.float(), b.float(), float32_done)

    def float64_done(h):
        stalled = (
            len(h) >= STALL_CHUNKS
            and h[-STALL_CHUNKS][0] - h[-1][0] < STALL_TOL
        )
        return (
            h[-1][1] < GRAD_TOL
            or stalled
            or len(h) * CHUNK_ITER >= MAX_ITER_FP64
        )

    W, b, iters64, grad = lbfgs_phase(X64, y, l2, W.double(), b.double(), float64_done)

    return W, b, iters32 + iters64, grad < GRAD_TOL, grad


def probe(X, y):
    """Fit the L2 grid on train (warm-started), pick by val accuracy, report on test."""

    X_train, X_val, X_test = [
        t.to(lp.DEVICE, torch.float64)
        for t in lp.standardize(X["train"], X["val"], X["test"])
    ]
    X_train32 = X_train.float()
    y_train = torch.as_tensor(y["train"], device=lp.DEVICE)
    num_classes = int(y_train.max()) + 1

    W = torch.zeros(num_classes, X_train.shape[1], device=lp.DEVICE, dtype=torch.float64)
    b = torch.zeros(num_classes, device=lp.DEVICE, dtype=torch.float64)

    best = None

    for l2 in L2_GRID:
        start = time.perf_counter()
        W, b, iterations, converged, grad = fit_logreg(
            X_train32, X_train, y_train, l2, W, b,
        )

        with torch.no_grad():
            val_pred = (X_val @ W.T + b).argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y["val"], val_pred)
        val_f1 = f1_score(y["val"], val_pred, average="macro")

        print(
            f"  l2={l2:<8g} iters={iterations:<6d} max|grad|={grad:.1e}"
            f"{'' if converged else ' (NOT converged)'} "
            f"val acc={val_acc:.4f} val Macro-F1={val_f1:.4f} "
            f"({time.perf_counter() - start:.1f}s)"
        )

        if best is None or (val_acc, val_f1) > (best["val_acc"], best["val_f1"]):
            with torch.no_grad():
                test_pred = (X_test @ W.T + b).argmax(dim=1).cpu().numpy()
            best = {
                "l2": l2,
                "converged": converged,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_acc": accuracy_score(y["test"], test_pred),
                "test_f1": f1_score(y["test"], test_pred, average="macro"),
            }

    print(f"  Selected l2={best['l2']:g} (best val accuracy)")
    return best


def append_result(row):
    new_file = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "encoders",
        nargs="+",
        help='"random", MIM checkpoint paths and/or "timm:<model name>"',
    )
    args = parser.parse_args()

    splits = load_splits()
    y = {split: df["label"].to_numpy() for split, df in splits.items()}

    print("RESISC45:", {s: len(df) for s, df in splits.items()},
          "|", len(np.unique(y["train"])), "classes")

    summary = []

    for source in args.encoders:
        print("\n" + "=" * 70 + f"\n{source}\n" + "=" * 70)
        features = extract(source, splits)

        for pooling in ("cls", "mean"):
            print(f"\nPooling: {pooling}")
            result = probe({s: features[s][pooling] for s in SPLITS}, y)
            summary.append((source, pooling, result))

            append_result({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "benchmark": "resisc45",
                "encoder": source,
                "pooling": pooling,
                "probe": "logreg_lbfgs_gpu",
                "l2": f"{result['l2']:g}",
                "converged": result["converged"],
                "val_accuracy": f"{result['val_acc']:.4f}",
                "val_macro_f1": f"{result['val_f1']:.4f}",
                "test_accuracy": f"{result['test_acc']:.4f}",
                "test_macro_f1": f"{result['test_f1']:.4f}",
            })

    print("\n" + "=" * 70 + "\nRESISC45 LINEAR PROBE (test)\n" + "=" * 70)
    print(f"{'Encoder':55s}{'Accuracy':>10s}{'Macro-F1':>10s}")
    for source, pooling, result in summary:
        print(f"{source + ' (' + pooling + ')':55s}"
              f"{result['test_acc']:>10.4f}{result['test_f1']:>10.4f}")
    print(f"\nResults appended to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
