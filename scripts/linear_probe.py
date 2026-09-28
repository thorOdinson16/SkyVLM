"""
Linear-probe evaluation of frozen ViT encoders on SkyScript categories.

Uses a fixed geographic split (see make_probe_split.py): the probe is
trained on split == "train" and evaluated on --eval-split (val by default).
Test should only be used once the masking ratio and probe settings are final.

Examples:
    python linear_probe.py random checkpoints/mim/mim_epoch_01.pt
    python linear_probe.py random checkpoints/mim/main/mim_latest.pt --eval-split test
    python linear_probe.py random --split-file dataset/manifests/probe_geo_split_plain.csv
"""

import argparse
import os
import csv
import hashlib
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from vit import SmallViT


# ---------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SPLIT_FILE = ROOT / "dataset" / "manifests" / "probe_geo_split.csv"

FEATURE_DIR = (
    ROOT
    / "checkpoints"
    / "eval"
)

RESULTS_FILE = FEATURE_DIR / "probe_results.csv"

IMAGE_SIZE = 224

BATCH_SIZE = 64

# Overridable so the pipeline can reduce RAM use
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))

# Probe settings: identical for every encoder
PROBE_BATCH_SIZE = 512

PROBE_EPOCHS = 5

PROBE_LR = 1e-3

PROBE_WEIGHT_DECAY = 1e-4

PROBE_SEED = 0

RANDOM_ENCODER_SEED = 0

# Encoders given as "timm:<model name>" are loaded from timm (reference
# baselines, e.g. an ImageNet-pretrained ViT-S/16)
TIMM_PREFIX = "timm:"

# Full probe: logistic regression solved to convergence with Newton's
# method, L2 strength selected on val. (train_probe below, a short SGD
# probe, is only used by the quick probe during MIM training.)
PROBE_METHOD = "logreg_newton"
L2_GRID = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 0.0]
NEWTON_MAX_ITER = 50
NEWTON_GRAD_TOL = 1e-7
NEWTON_JITTER = 1e-8

# Longitude bands: < -30, -30..60, > 60
REGIONS = ["Americas", "Europe/Africa", "Asia/Oceania"]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class SkyScriptDataset(Dataset):

    def __init__(self, df):

        self.df = df.reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = (
            ROOT
            / "dataset"
            / "images"
            / row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        return image, int(row["label"])


def load_split(split_file):
    """Read a probe split file and attach integer labels."""

    df = pd.read_csv(
        split_file,
        keep_default_na=False,
    )

    classes = sorted(
        df["category"]
        .unique()
        .tolist()
    )

    df["label"] = df["category"].map(
        {category: i for i, category in enumerate(classes)}
    )

    # Coarse region from the cell's longitude, for per-region reporting
    lon = df["geo_cell_1deg"].str.split("_").str[1].astype(int)

    df["region"] = pd.cut(
        lon,
        [-181, -30, 60, 181],
        labels=REGIONS,
    ).astype(str)

    return df, classes


# ---------------------------------------------------------
# Load ViT
# ---------------------------------------------------------

class TimmEncoder(nn.Module):
    """
    Wraps a pretrained timm ViT so it takes the same [0, 1] resized images
    as our encoders and returns [CLS, patch tokens] like SmallViT.
    """

    def __init__(self, name):
        super().__init__()

        import timm

        self.model = timm.create_model(name, pretrained=True, num_classes=0)

        config = timm.data.resolve_data_config({}, model=self.model)
        self.register_buffer("mean", torch.tensor(config["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config["std"]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model.forward_features((x - self.mean) / self.std)


def load_timm_encoder(name):

    print(f"Encoder: timm pretrained {name}")

    model = TimmEncoder(name).to(DEVICE)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def load_encoder(checkpoint=None):

    if checkpoint is None:

        print(
            "Encoder: randomly initialized"
        )

        # Fixed seed so the random baseline is reproducible
        torch.manual_seed(RANDOM_ENCODER_SEED)

        model = SmallViT()

    else:

        print(
            f"Encoder: {checkpoint}"
        )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        config = state.get("config", {})

        model = SmallViT(
            img_size=config.get("image_size", 224),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 512),
            depth=config.get("depth", 8),
            num_heads=config.get("num_heads", 8),
            mlp_dim=config.get("mlp_dim", 2048),
        )

        # train_mim.py saves the full MIMModel; the encoder
        # weights live under "model_state_dict" with a "vit." prefix.
        encoder_state = {
            key[len("vit."):]: value
            for key, value in state["model_state_dict"].items()
            if key.startswith("vit.")
        }

        model.load_state_dict(
            encoder_state,
            strict=True,
        )

    model = model.to(DEVICE)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def images_fingerprint(df):
    return hashlib.sha1(
        "\n".join(df["image_path"]).encode()
    ).hexdigest()


def cache_path_for(source):
    name = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_")
    return FEATURE_DIR / f"features_{name}.pt"


@torch.inference_mode()
def extract_features(
    model,
    df,
    cache_path,
    source,
):

    fingerprint = images_fingerprint(df)

    # Cached features are only reused if they came from the same
    # encoder file and cover exactly the same images in the same order.
    if cache_path.exists():

        data = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        if (
            data.get("source") == source
            and data.get("images") == fingerprint
        ):

            print(
                f"\nLoading cached features:"
                f"\n{cache_path}"
            )

            return {
                "cls": data["cls"],
                "mean": data["mean"],
            }

        print(
            f"\nIgnoring stale feature cache:"
            f"\n{cache_path}"
        )

    loader = DataLoader(
        SkyScriptDataset(df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print("\nExtracting features...")
    print(
        f"Images: {len(df):,}"
    )

    features = {"cls": [], "mean": []}

    start = time.perf_counter()

    for images, _ in tqdm(
        loader,
        desc="Feature extraction",
        unit="batch",
    ):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        output = model(images)

        # CLS token, and the mean over patch tokens
        features["cls"].append(
            output[:, 0, :].cpu()
        )

        features["mean"].append(
            output[:, 1:, :].mean(dim=1).cpu()
        )

    features = {
        pooling: torch.cat(chunks)
        for pooling, chunks in features.items()
    }

    elapsed = time.perf_counter() - start

    print(
        f"\nExtraction time: "
        f"{elapsed / 60:.2f} minutes "
        f"({len(df) / elapsed:.0f} images/sec)"
    )

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            **features,
            "source": source,
            "images": fingerprint,
        },
        cache_path,
    )

    print(
        f"Saved features:"
        f"\n{cache_path}"
    )

    return features


# ---------------------------------------------------------
# Linear probe
# ---------------------------------------------------------

def standardize(X_train, *others):
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-6
    return [(X - mean) / std for X in (X_train, *others)]


def fit_logreg_newton(X, y, num_classes, l2, W):
    """
    Multinomial logistic regression (mean cross-entropy + 0.5 * l2 * |W|^2,
    bias unregularized) solved with Newton's method and a backtracking line
    search. X already has a trailing column of ones for the bias. Converges
    to numerical precision in a few steps even with highly correlated
    features, unlike first-order methods.
    """

    n, dim = X.shape
    Y = nn.functional.one_hot(y, num_classes).to(X.dtype)

    reg = torch.ones(dim, device=X.device, dtype=torch.float64)
    reg[-1] = 0.0                       # no penalty on the bias
    reg_flat = reg.repeat(num_classes)

    def objective(W):
        logits = X @ W.T
        loss = nn.functional.cross_entropy(logits, y)
        return loss + 0.5 * l2 * (W[:, :-1].double() ** 2).sum()

    loss = objective(W).item()

    for iteration in range(1, NEWTON_MAX_ITER + 1):

        P = torch.softmax(X @ W.T, dim=1)

        grad = ((P - Y).T @ X / n).double()
        grad[:, :-1] += l2 * W[:, :-1].double()

        if grad.abs().max().item() < NEWTON_GRAD_TOL:
            return W, loss, iteration - 1, True

        # Hessian blocks: X^T diag(p_c (delta_cd - p_d)) X / n
        H = torch.empty(
            num_classes * dim, num_classes * dim,
            device=X.device, dtype=torch.float64,
        )
        for c in range(num_classes):
            for d in range(c, num_classes):
                weight = P[:, c] * (float(c == d) - P[:, d])
                block = ((X * weight[:, None]).T @ X / n).double()
                H[c*dim:(c+1)*dim, d*dim:(d+1)*dim] = block
                H[d*dim:(d+1)*dim, c*dim:(c+1)*dim] = block.T

        # L2 on weights; tiny jitter on the bias directions, which are
        # otherwise singular (softmax is shift-invariant)
        H += torch.diag(l2 * reg_flat + NEWTON_JITTER)

        step = torch.linalg.solve(H, grad.reshape(-1)).reshape(num_classes, dim)

        # Backtracking line search
        t = 1.0
        while True:
            W_new = (W.double() - t * step).to(X.dtype)
            new_loss = objective(W_new).item()
            if new_loss <= loss - 1e-4 * t * (grad * step).sum().item() or t < 1e-6:
                break
            t *= 0.5

        improvement = loss - new_loss
        W, loss = W_new, new_loss

        if improvement < 1e-12:
            return W, loss, iteration, True

    return W, loss, NEWTON_MAX_ITER, False


def logreg_probe(
    X_train,
    y_train,
    X_val,
    y_val,
    X_eval,
    num_classes,
):
    """
    Logistic-regression probe: for each L2 strength in L2_GRID (strongest
    first, warm-started) fit to convergence on train, pick the strength
    with the best val Macro-F1, and return its predictions on X_eval.
    Deterministic: zero init, full-batch Newton.
    """

    def with_bias(X):
        X = X.to(DEVICE, torch.float32)
        return torch.cat([X, torch.ones(len(X), 1, device=DEVICE)], dim=1)

    X_train, X_val, X_eval = [
        with_bias(X)
        for X in standardize(X_train, X_val, X_eval)
    ]
    y_train = y_train.to(DEVICE)

    W = torch.zeros(num_classes, X_train.shape[1], device=DEVICE)

    best = None

    for l2 in L2_GRID:

        W, loss, iterations, converged = fit_logreg_newton(
            X_train, y_train, num_classes, l2, W,
        )

        val_pred = (X_val @ W.T).argmax(dim=1).cpu()

        val_f1 = f1_score(
            y_val,
            val_pred,
            labels=range(num_classes),
            average="macro",
            zero_division=0,
        )
        val_acc = accuracy_score(y_val, val_pred)

        print(
            f"  l2={l2:<8g} train loss={loss:.5f} newton iters={iterations:<3d}"
            f"{'' if converged else ' (NOT converged)'} "
            f"val Macro-F1={val_f1:.4f} val acc={val_acc:.4f}"
        )

        if best is None or (val_f1, val_acc) > (best["val_f1"], best["val_acc"]):
            best = {
                "l2": l2,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "converged": converged,
                "eval_pred": (X_eval @ W.T).argmax(dim=1).cpu(),
            }

    print(f"  Selected l2={best['l2']:g} (best val Macro-F1)")

    return best


def train_probe(*args):

    # Same seed for every encoder: identical classifier init and
    # batch order. fork_rng keeps this from resetting the caller's
    # RNG (e.g. MIM masking when the quick probe runs mid-training).
    devices = [torch.cuda.current_device()] if DEVICE.type == "cuda" else []

    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(PROBE_SEED)
        return _train_probe(*args)


def _train_probe(
    X_train,
    y_train,
    X_test,
    y_test,
    num_classes,
):

    # Standardize with training-set statistics so CLS and
    # mean-pooled features are compared on equal footing.
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-6

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    classifier = nn.Linear(
        X_train.shape[1],
        num_classes,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=PROBE_LR,
        weight_decay=PROBE_WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(
            X_train,
            y_train,
        ),
        batch_size=PROBE_BATCH_SIZE,
        shuffle=True,
    )

    for epoch in range(PROBE_EPOCHS):

        classifier.train()

        total_loss = 0.0

        for X, y in train_loader:

            X = X.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = criterion(
                classifier(X),
                y,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"  Probe epoch {epoch + 1}/{PROBE_EPOCHS}: "
            f"loss={total_loss / len(train_loader):.4f}"
        )

    classifier.eval()

    with torch.inference_mode():

        predictions = (
            classifier(X_test.to(DEVICE))
            .argmax(dim=1)
            .cpu()
        )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    return (
        accuracy,
        macro_f1,
        predictions,
    )


# ---------------------------------------------------------
# Experiment
# ---------------------------------------------------------

def run_experiment(
    source,
    df,
    classes,
    eval_split,
    split_file,
):

    print("\n")
    print("=" * 70)
    print(source)
    print("=" * 70)

    if source.startswith(TIMM_PREFIX):

        model = load_timm_encoder(source[len(TIMM_PREFIX):])
        cache_source = source

    else:

        checkpoint = None if source == "random" else Path(source)

        model = load_encoder(checkpoint)

        # Identify checkpoints by path + modification time, so a
        # retrained checkpoint at the same path is not served from cache.
        cache_source = (
            "random"
            if checkpoint is None
            else f"{checkpoint.resolve()}:{checkpoint.stat().st_mtime_ns}"
        )

    features = extract_features(
        model,
        df,
        cache_path_for(source),
        cache_source,
    )

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    train_mask = torch.tensor((df["split"] == "train").values)
    val_mask = torch.tensor((df["split"] == "val").values)
    eval_mask = torch.tensor((df["split"] == eval_split).values)

    labels = torch.tensor(df["label"].values)

    results = {}

    for pooling, pooled in features.items():

        print(f"\nPooling: {pooling}")

        # L2 strength is always chosen on val, whatever the eval split
        probe = logreg_probe(
            pooled[train_mask],
            labels[train_mask],
            pooled[val_mask],
            labels[val_mask],
            pooled[eval_mask],
            len(classes),
        )

        accuracy, macro_f1 = report_predictions(
            source=source,
            df=df,
            classes=classes,
            split_file=split_file,
            eval_split=eval_split,
            predictions=probe["eval_pred"],
            pooling=pooling,
            method=PROBE_METHOD,
            details={"l2": f"{probe['l2']:g}", "converged": probe["converged"]},
        )

        results[pooling] = (accuracy, macro_f1)

    return results


def report_predictions(
    source,
    df,
    classes,
    split_file,
    eval_split,
    predictions,
    pooling,
    method,
    details,
):
    """
    Print and record overall and per-region Macro-F1 / accuracy for
    predictions on df[split == eval_split]. The primary split's val/test
    lean towards the Americas, so regions are reported separately too.
    """

    eval_mask = (df["split"] == eval_split).values
    y_eval = df.loc[eval_mask, "label"].values
    eval_regions = df.loc[eval_mask, "region"].values
    predictions = predictions.numpy()

    def scores(keep):
        return (
            accuracy_score(y_eval[keep], predictions[keep]),
            f1_score(
                y_eval[keep],
                predictions[keep],
                labels=range(len(classes)),
                average="macro",
                zero_division=0,
            ),
        )

    everything = y_eval == y_eval
    accuracy, macro_f1 = scores(everything)

    print(
        f"\n{eval_split} Macro-F1 : {macro_f1:.4f}"
        f"\n{eval_split} Accuracy : {accuracy:.4f}"
    )

    print("\nClassification report")
    print("-" * 70)

    print(
        classification_report(
            y_eval,
            predictions,
            labels=range(len(classes)),
            target_names=classes,
            digits=4,
            zero_division=0,
        )
    )

    breakdown = [("all", everything)] + [
        (region, eval_regions == region)
        for region in REGIONS
        if (eval_regions == region).any()
    ]

    print(f"{'region':16s}{'images':>8s}{'Macro-F1':>10s}{'Accuracy':>10s}")

    for region, keep in breakdown:

        region_acc, region_f1 = scores(keep)

        print(f"{region:16s}{int(keep.sum()):>8,d}{region_f1:>10.4f}{region_acc:>10.4f}")

        append_result(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "encoder": source,
                "split_file": Path(split_file).name,
                "eval_split": eval_split,
                "pooling": pooling,
                "probe": method,
                "l2": details.get("l2", ""),
                "converged": details.get("converged", ""),
                "region": region,
                "macro_f1": f"{region_f1:.4f}",
                "accuracy": f"{region_acc:.4f}",
                "train_images": int((df["split"] == "train").sum()),
                "eval_images": int(keep.sum()),
            }
        )

    return accuracy, macro_f1


def append_result(row):
    new_file = not RESULTS_FILE.exists()

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "encoders",
        nargs="+",
        help='"random", MIM checkpoint paths and/or "timm:<model name>"',
    )
    parser.add_argument(
        "--split-file",
        default=str(SPLIT_FILE),
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    print("=" * 70)
    print("SkyScript Linear Probe Evaluation")
    print("=" * 70)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    df, classes = load_split(args.split_file)

    print("Split file:", args.split_file)
    print("Evaluating on:", args.eval_split)
    print("Classes:", classes)

    print(
        f"Train images: {(df['split'] == 'train').sum():,}"
    )
    print(
        f"{args.eval_split} images: {(df['split'] == args.eval_split).sum():,}"
    )

    all_results = {
        source: run_experiment(
            source,
            df,
            classes,
            args.eval_split,
            args.split_file,
        )
        for source in args.encoders
    }

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print(f"FINAL COMPARISON ({args.eval_split})")
    print("=" * 70)

    print(
        f"{'Encoder':50s}"
        f"{'Macro-F1':>10s}"
        f"{'Accuracy':>10s}"
    )

    for source, results in all_results.items():
        for pooling, (accuracy, macro_f1) in results.items():
            print(
                f"{source + ' (' + pooling + ')':50s}"
                f"{macro_f1:>10.4f}"
                f"{accuracy:>10.4f}"
            )

    print(f"\nResults appended to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
