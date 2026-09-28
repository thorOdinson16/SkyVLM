from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

OUTPUT = Path("dataset/text/fineweb_edu")

TARGET_DOCUMENTS = 1_000_000

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = OUTPUT / "fineweb_edu_1m.txt"


# ---------------------------------------------------------
# Load FineWeb-Edu in streaming mode
# ---------------------------------------------------------

print("=" * 70)
print("FineWeb-Edu Streaming Downloader")
print("=" * 70)

print(
    f"Target documents: {TARGET_DOCUMENTS:,}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print("\nLoading dataset...")

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

print("Dataset loaded.")
print("Streaming documents...\n")


# ---------------------------------------------------------
# Write documents
# ---------------------------------------------------------

count = 0

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    progress = tqdm(
        total=TARGET_DOCUMENTS,
        desc="Downloading",
        unit="docs",
    )

    for example in dataset:

        text = example.get("text")

        if not text:
            continue

        text = text.strip()

        if len(text) < 100:
            continue

        f.write(text)
        f.write("\n\n")

        count += 1

        progress.update(1)

        if count >= TARGET_DOCUMENTS:
            break

    progress.close()


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Documents saved: {count:,}"
)

print(
    f"Output file: {OUTPUT_FILE}"
)