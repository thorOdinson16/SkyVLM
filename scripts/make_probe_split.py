"""
Fixed geographic splits for linear-probe evaluation.

Images are grouped into 1-degree lat/lon grid cells and every cell is
assigned to exactly one of train / val / test (a geographically disjoint
grid-cell split). Note that neighbouring cells share a boundary, so this
does NOT guarantee any minimum distance between train and test images.

Two variants are written:

  probe_geo_split.csv        PRIMARY. Dense cells (>= 0.5% of all images)
                             are always assigned to train; the remaining
                             cells are split ~70/10/20 overall. This stops a
                             few dense city clusters from dominating val/test,
                             but means val/test measure generalization to
                             less densely sampled regions.

  probe_geo_split_plain.csv  SENSITIVITY ANALYSIS. All cells split 7/1/2 over
                             10 folds with no dense-cell rule.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "selected_200k_final.csv"
MANIFEST_DIR = ROOT / "dataset" / "manifests"

# 1-degree cells. The 5-degree "geo_cell" in the source is too
# coarse: one cell holds 26% of all images.
GRID_DEGREES = 1.0

# Primary split: cells holding >= this share of all images go to train
DENSE_CELL_THRESHOLD = 0.005
DENSE_FOLDS = 20

# Plain split: 10 folds -> 7 train / 1 val / 2 test
PLAIN_FOLDS = 10

TARGET_VAL = 0.10
TARGET_TEST = 0.20

# Fixed subset used by the quick probe during MIM training
QUICK_TRAIN = 16_000
QUICK_VAL = 4_000

SEED = 42


# ---------------------------------------------------------
# Split construction
# ---------------------------------------------------------

def load_cells():
    df = pd.read_csv(
        SOURCE,
        usecols=[
            "filepath",
            "category",
            "latitude",
            "longitude",
        ],
    )

    lat_cell = np.floor(df["latitude"] / GRID_DEGREES) * GRID_DEGREES
    lon_cell = np.floor(df["longitude"] / GRID_DEGREES) * GRID_DEGREES

    df["geo_cell_1deg"] = (
        lat_cell.astype(int).astype(str)
        + "_"
        + lon_cell.astype(int).astype(str)
    )

    return df


def assign_folds(df, num_folds, num_val, num_test):
    """Group folds by cell; first folds -> val, next -> test, rest -> train."""

    split = pd.Series("train", index=df.index)

    folds = StratifiedGroupKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=SEED,
    )

    for fold, (_, idx) in enumerate(
        folds.split(df, df["category"], df["geo_cell_1deg"])
    ):
        if fold < num_val:
            split[df.index[idx]] = "val"
        elif fold < num_val + num_test:
            split[df.index[idx]] = "test"

    return split


def plain_split(df):
    df = df.copy()
    df["dense_cell"] = False

    # Folds 0-6 train, 7 val, 8-9 test
    folds = StratifiedGroupKFold(
        n_splits=PLAIN_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    df["split"] = "test"

    for fold, (_, idx) in enumerate(
        folds.split(df, df["category"], df["geo_cell_1deg"])
    ):
        if fold < 7:
            df.loc[df.index[idx], "split"] = "train"
        elif fold == 7:
            df.loc[df.index[idx], "split"] = "val"

    return df


def dense_cell_split(df):
    df = df.copy()

    counts = df["geo_cell_1deg"].value_counts()
    dense = counts[counts >= DENSE_CELL_THRESHOLD * len(df)].index

    df["dense_cell"] = df["geo_cell_1deg"].isin(dense)

    rest = df[~df["dense_cell"]]
    rest_share = len(rest) / len(df)

    # Fold counts chosen so val/test hit their overall targets
    num_val = round(TARGET_VAL / rest_share * DENSE_FOLDS)
    num_test = round(TARGET_TEST / rest_share * DENSE_FOLDS)

    df["split"] = "train"
    df.loc[rest.index, "split"] = assign_folds(
        rest,
        DENSE_FOLDS,
        num_val,
        num_test,
    )

    return df


def add_quick_probe(df):
    df["quick_probe"] = ""

    for split, size in (("train", QUICK_TRAIN), ("val", QUICK_VAL)):
        pool = df[df["split"] == split]
        chosen, _ = train_test_split(
            pool.index,
            train_size=size,
            random_state=SEED,
            stratify=pool["category"],
        )
        df.loc[chosen, "quick_probe"] = split

    return df


def save(df, path):
    df = df.rename(columns={"filepath": "image_path"})[
        [
            "image_path",
            "category",
            "geo_cell_1deg",
            "dense_cell",
            "split",
            "quick_probe",
        ]
    ]

    df.to_csv(path, index=False)
    print("Saved:", path)


def main():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    df = load_cells()

    save(
        add_quick_probe(dense_cell_split(df)),
        MANIFEST_DIR / "probe_geo_split.csv",
    )

    save(
        add_quick_probe(plain_split(df)),
        MANIFEST_DIR / "probe_geo_split_plain.csv",
    )


if __name__ == "__main__":
    main()
