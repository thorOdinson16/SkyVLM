from pathlib import Path
import random

import sentencepiece as spm


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

FINEWEB = (
    ROOT
    / "dataset"
    / "text"
    / "fineweb_edu"
    / "fineweb_edu_1m.txt"
)

SKYSCRIPT = (
    ROOT
    / "dataset"
    / "manifests"
    / "train_200k.csv"
)

TOKENIZER_DIR = (
    ROOT
    / "dataset"
    / "tokenizer"
)

TOKENIZER_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TRAIN_TEXT = TOKENIZER_DIR / "tokenizer_training.txt"

MODEL_PREFIX = TOKENIZER_DIR / "skyvlm"

VOCAB_SIZE = 16_384

# Number of FineWeb characters used for tokenizer training.
# This is only for training the tokenizer, NOT LM training.
FINEWEB_CHARS = 300_000_000

RANDOM_SEED = 42


# =========================================================
# Prepare tokenizer training text
# =========================================================

def prepare_training_text():

    print("=" * 70)
    print("Preparing tokenizer training corpus")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    if TRAIN_TEXT.exists():
        print(
            f"Existing tokenizer corpus found:\n"
            f"{TRAIN_TEXT}"
        )
        return

    print("Reading FineWeb-Edu...")

    with open(
        FINEWEB,
        "r",
        encoding="utf-8",
    ) as f:

        text = f.read(
            FINEWEB_CHARS
        )

    print(
        f"FineWeb characters: "
        f"{len(text):,}"
    )

    # -----------------------------------------------------
    # Add SkyScript captions
    # -----------------------------------------------------

    print("Reading SkyScript captions...")

    import pandas as pd

    df = pd.read_csv(
        SKYSCRIPT,
        usecols=["caption"],
    )

    captions = (
        df["caption"]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(
        f"SkyScript captions: "
        f"{len(captions):,}"
    )

    with open(
        TRAIN_TEXT,
        "w",
        encoding="utf-8",
    ) as out:

        out.write(text)

        out.write("\n\n")

        for caption in captions:
            out.write(caption)
            out.write("\n")

    print(
        f"\nTokenizer training file:\n"
        f"{TRAIN_TEXT}"
    )

    print(
        f"File size: "
        f"{TRAIN_TEXT.stat().st_size / 1024 / 1024:.2f} MB"
    )


# =========================================================
# Train SentencePiece
# =========================================================

def train_tokenizer():

    print("\n" + "=" * 70)
    print("Training SentencePiece BPE tokenizer")
    print("=" * 70)

    model_file = (
        MODEL_PREFIX.with_suffix(".model")
    )

    if model_file.exists():

        print(
            f"Tokenizer already exists:\n"
            f"{model_file}"
        )

        return

    spm.SentencePieceTrainer.train(

        input=str(TRAIN_TEXT),

        model_prefix=str(MODEL_PREFIX),

        vocab_size=VOCAB_SIZE,

        model_type="bpe",

        character_coverage=1.0,

        normalization_rule_name="nmt_nfkc",

        split_digits=True,

        byte_fallback=True,

        add_dummy_prefix=True,

        unk_id=0,

        bos_id=1,

        eos_id=2,

        pad_id=3,

        user_defined_symbols=[
            "<image>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ],

        max_sentence_length=8192,

        input_sentence_size=2_000_000,

        shuffle_input_sentence=True,

    )

    print(
        f"\nTokenizer saved:\n"
        f"{model_file}"
    )


# =========================================================
# Test tokenizer
# =========================================================

def test_tokenizer():

    print("\n" + "=" * 70)
    print("Testing tokenizer")
    print("=" * 70)

    model_file = (
        MODEL_PREFIX.with_suffix(".model")
    )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(model_file)
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size(),
    )

    examples = [
        "A residential area with several buildings and roads.",
        "Satellite imagery shows agricultural fields.",
        "A large highway intersects an urban settlement.",
        "Remote sensing image of buildings near a river.",
        "power utilities and transportation infrastructure",
    ]

    for text in examples:

        pieces = tokenizer.encode(
            text,
            out_type=str,
        )

        ids = tokenizer.encode(
            text,
            out_type=int,
        )

        print("\nText:")
        print(text)

        print(
            "Tokens:",
            pieces,
        )

        print(
            "Token count:",
            len(ids),
        )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 70)
    print("SkyVLM Tokenizer")
    print("=" * 70)

    print(
        "FineWeb:",
        FINEWEB,
    )

    print(
        "SkyScript:",
        SKYSCRIPT,
    )

    print(
        "Vocabulary:",
        VOCAB_SIZE,
    )

    prepare_training_text()

    train_tokenizer()

    test_tokenizer()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()