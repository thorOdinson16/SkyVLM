"""
Image <-> text retrieval on the official SkyScript val/test captions.

    python scripts/eval_retrieval.py skyclip --split val --n 1000
    python scripts/eval_retrieval.py ours --clip-ckpt checkpoints/clip/main/clip_latest.pt --split val --n 1000

Same protocol for every model: the first N pairs of the split are embedded; each
image ranks the N captions (i2t) and each caption ranks the N images (t2i). A hit
means the retrieved caption has the *same text* as the ground truth (templated
captions repeat, so index equality alone would undercount).

skyclip : SkyCLIP ViT-B/32 (SkyScript authors' CLIP, trained on this dataset), via open_clip.
          Captions beyond its 77-token context are truncated.
ours    : our ViT + LM text tower from clip_align.py.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "checkpoints" / "eval" / "retrieval"
IMAGE_DIR = ROOT / "dataset" / "images"
EXTERNAL = ROOT / "dataset" / "external"


def load_pairs(split, n):
    name = {"val": "val_5k.csv", "test": "test_30k.csv"}[split]
    df = pd.read_csv(ROOT / "dataset" / "manifests" / name, keep_default_na=False)
    df = df[df["caption"].str.strip() != ""].reset_index(drop=True).iloc[:n]
    return df["image_path"].tolist(), df["caption"].str.strip().tolist()


def metrics(img, txt, captions):
    """img, txt: L2-normalized [N, D] tensors on DEVICE."""

    ids = {}
    key = torch.tensor([ids.setdefault(c, len(ids)) for c in captions], device=img.device)
    match = key[:, None] == key[None, :]
    sim = img @ txt.t()

    out = {"n": len(captions), "unique_captions": len(ids)}
    for name, s, m in [("i2t", sim, match), ("t2i", sim.t(), match.t())]:
        order = s.argsort(dim=1, descending=True)
        hits = m.gather(1, order)
        for k in (1, 5, 10):
            out[f"{name}_R@{k}"] = round(hits[:, :k].any(dim=1).float().mean().item(), 4)

    return out


@torch.no_grad()
def embed_skyclip(paths, captions, batch):
    import open_clip

    ckpt_dir = EXTERNAL / "SkyCLIP_ViT_B32_top50pct"
    if not ckpt_dir.exists():
        with zipfile.ZipFile(EXTERNAL / "SkyCLIP_ViT_B32_top50pct.zip") as z:
            z.extractall(EXTERNAL)

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32")

    # The checkpoint holds a NumPy scalar that torch's safe loader rejects by default;
    # allowlist exactly that instead of loading with weights_only=False.
    import numpy as np

    def alias(module, name, real):   # forwards to the real function under the pickle's (old) NumPy path
        def f(*args):
            return real(*args)
        f.__module__, f.__name__, f.__qualname__ = module, name, name
        return f

    scalar = (np._core if hasattr(np, "_core") else np.core).multiarray.scalar

    allowed = [alias("numpy.core.multiarray", "scalar", scalar), np.dtype, type(np.dtype("float64"))]

    with torch.serialization.safe_globals(allowed):
        ckpt = torch.load(ckpt_dir / "epoch_20.pt", map_location="cpu", weights_only=True)

    state = ckpt.get("state_dict", ckpt)
    state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    print(model.load_state_dict(state, strict=True), flush=True)
    del ckpt, state

    model = model.to(DEVICE).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    imgs, txts = [], []
    for i in range(0, len(paths), batch):
        x = torch.stack([
            preprocess(Image.open(IMAGE_DIR / p).convert("RGB")) for p in paths[i:i + batch]
        ]).to(DEVICE)
        t = tokenizer(captions[i:i + batch]).to(DEVICE)

        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            imgs.append(torch.nn.functional.normalize(model.encode_image(x).float(), dim=-1))
            txts.append(torch.nn.functional.normalize(model.encode_text(t).float(), dim=-1))

    return torch.cat(imgs), torch.cat(txts), sum(p.numel() for p in model.parameters())


@torch.no_grad()
def embed_ours(paths, captions, batch, clip_ckpt):
    import sentencepiece as spm
    from torchvision import transforms

    from clip_align import MAX_TEXT_TOKENS, ClipModel
    from vlm import IMAGE_SIZE, TOKENIZER_PATH, load_lm, load_vit

    tok = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))

    model = ClipModel(load_vit(), load_lm())
    state = torch.load(clip_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(DEVICE).eval()

    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

    imgs, txts = [], []
    for i in range(0, len(paths), batch):
        x = torch.stack([tf(Image.open(IMAGE_DIR / p).convert("RGB")) for p in paths[i:i + batch]]).to(DEVICE)

        seqs = [tok.encode(c)[: MAX_TEXT_TOKENS - 1] + [tok.eos_id()] for c in captions[i:i + batch]]
        lengths = torch.tensor([len(s) for s in seqs], device=DEVICE)
        ids = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long, device=DEVICE)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            imgs.append(model.encode_image(x))
            txts.append(model.encode_text(ids, lengths))

    return torch.cat(imgs), torch.cat(txts), sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["skyclip", "ours"])
    p.add_argument("--clip-ckpt", help="clip_align.py state (clip_latest.pt) for 'ours'")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--tag")
    args = p.parse_args()

    paths, captions = load_pairs(args.split, args.n)

    if args.model == "skyclip":
        img, txt, n_params = embed_skyclip(paths, captions, args.batch)
    else:
        if not args.clip_ckpt:
            p.error("--clip-ckpt is required for 'ours'")
        img, txt, n_params = embed_ours(paths, captions, args.batch, args.clip_ckpt)

    result = metrics(img, txt, captions)
    result.update({"model": args.model, "split": args.split, "params_M": round(n_params / 1e6, 1)})
    print(json.dumps(result), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.model}_{args.split}_{args.n}"
    with open(OUT_DIR / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
