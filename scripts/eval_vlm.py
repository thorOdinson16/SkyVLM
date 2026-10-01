"""
Caption evaluation of a trained SkyVLM on the official SkyScript val/test captions.

    python scripts/eval_vlm.py --ckpt checkpoints/vlm/stage2/vlm_final.pt --split val --n 1000
    python scripts/eval_vlm.py --ckpt ... --split test --n 5000 --controls

Greedy decoding. Metrics: BLEU-4, ROUGE-L, CIDEr-D, plus two structure-aware
scores for the "a satellite image of <main>, surrounded by <obj>; <obj>" format:
main-object exact match and surrounding-object-set F1.

--controls also scores the same model with (a) images shuffled across the batch
and (b) a blank (zero) image; if the real-image scores are not clearly above
these, the model is not reading the image.
"""

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece as spm
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vlm import TOKENIZER_PATH, CaptionDataset, build_vlm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "checkpoints" / "eval" / "vlm"
PREFIX = "a satellite image of "


# ============================================================
# Metrics
# ============================================================

def tok(s):
    return s.lower().replace(",", " , ").replace(";", " ; ").split()


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = tok(pred), tok(ref)
    if not p or not r:
        return 0.0
    l = lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)


def ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def cider_d(preds, refs, sigma=6.0):
    """CIDEr-D with one reference per image; document frequencies come from
    the references of the evaluated set."""

    ref_toks = [tok(r) for r in refs]
    pred_toks = [tok(p) for p in preds]
    N = len(refs)

    df = [defaultdict(int) for _ in range(4)]
    for r in ref_toks:
        for n in range(4):
            for g in ngrams(r, n + 1):
                df[n][g] += 1

    log_n = math.log(float(N))

    def vec(words, n):
        counts = ngrams(words, n + 1)
        v = {g: c * (log_n - math.log(max(1.0, df[n][g]))) for g, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values()))
        return v, norm

    scores = []
    for p, r in zip(pred_toks, ref_toks):
        s = 0.0
        for n in range(4):
            vp, np_ = vec(p, n)
            vr, nr = vec(r, n)
            val = sum(min(x, vr[g]) * vr[g] for g, x in vp.items() if g in vr)
            if np_ and nr:
                val /= np_ * nr
            else:
                val = 0.0
            delta = len(p) - len(r)
            s += val * math.exp(-(delta ** 2) / (2 * sigma ** 2))
        scores.append(s / 4 * 10)

    return sum(scores) / len(scores)


def parse(caption):
    """-> (main object, set of surrounding objects)"""

    c = caption.strip().lower()
    if c.startswith(PREFIX):
        c = c[len(PREFIX):]

    main, _, rest = c.partition(", surrounded by ")
    objs = {o.strip() for o in rest.split(";") if o.strip()}

    return main.strip(), objs


def structure_scores(preds, refs):
    exact, f1s = 0, []

    for p, r in zip(preds, refs):
        pm, po = parse(p)
        rm, ro = parse(r)

        exact += pm == rm

        if not po and not ro:
            f1s.append(1.0)
        elif not po or not ro:
            f1s.append(0.0)
        else:
            tp = len(po & ro)
            prec, rec = tp / len(po), tp / len(ro)
            f1s.append(0.0 if tp == 0 else 2 * prec * rec / (prec + rec))

    return exact / len(refs), sum(f1s) / len(f1s)


def score(preds, refs):
    smooth = SmoothingFunction().method1
    bleu = corpus_bleu(
        [[tok(r)] for r in refs], [tok(p) for p in preds], smoothing_function=smooth
    )
    exact, f1 = structure_scores(preds, refs)

    return {
        "BLEU-4": round(bleu, 4),
        "ROUGE-L": round(sum(rouge_l(p, r) for p, r in zip(preds, refs)) / len(refs), 4),
        "CIDEr-D": round(cider_d(preds, refs), 4),
        "main_object_exact": round(exact, 4),
        "surrounding_F1": round(f1, 4),
    }


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_all(model, loader, tokenizer, mode, max_new_tokens):
    preds = []

    for images, _ in loader:
        images = images.to(DEVICE)

        if mode == "shuffled":
            images = images.roll(1, dims=0)
        elif mode == "blank":
            images = torch.zeros_like(images)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            out = model.generate(images, tokenizer.eos_id(), max_new_tokens)

        for row in out.tolist():
            if tokenizer.eos_id() in row:
                row = row[: row.index(tokenizer.eos_id())]
            preds.append(tokenizer.decode(row))

    return preds


def image_collate(batch):
    return torch.stack([b[0] for b in batch]), None


def merge(args):
    """Score the concatenated real-image samples of several shard runs."""

    preds, refs = [], []

    for tag in args.merge:
        with open(OUT_DIR / f"{tag}_{args.split}_samples.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                preds.append(row["pred"])
                refs.append(row["ref"])

    results = {"shards": args.merge, "split": args.split, "n": len(refs), "real": score(preds, refs)}
    tag = args.tag or "merged"
    print(f"[{tag}/{args.split}/real] n={len(refs)} {results['real']}", flush=True)

    with open(OUT_DIR / f"{tag}_{args.split}.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--start", type=int, default=0, help="first image index (for sharded runs)")
    p.add_argument("--merge", nargs="+", help="score the combined samples of these shard tags instead of generating")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--controls", action="store_true")
    p.add_argument("--tag", help="name for the output files (default: checkpoint stem)")
    args = p.parse_args()

    if args.merge:
        merge(args)
        return

    if not args.ckpt:
        p.error("--ckpt is required unless --merge is used")

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))

    model = build_vlm()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(DEVICE).eval()

    ds = CaptionDataset(
        f"{args.split}_{'5k' if args.split == 'val' else '30k'}.csv",
        tokenizer,
        limit=args.start + args.n,
    )
    ds.df = ds.df.iloc[args.start:].reset_index(drop=True)
    refs = ds.df["caption"].str.strip().tolist()

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=4, collate_fn=image_collate)

    modes = ["real"] + (["shuffled", "blank"] if args.controls else [])
    results = {"ckpt": args.ckpt, "step": state["step"], "split": args.split, "n": len(ds)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or Path(args.ckpt).stem

    for mode in modes:
        start = time.time()
        preds = generate_all(model, loader, tokenizer, mode, args.max_new_tokens)
        results[mode] = score(preds, refs)
        print(f"[{tag}/{args.split}/{mode}] {results[mode]}  ({time.time() - start:.0f}s)", flush=True)

        if mode == "real":
            with open(OUT_DIR / f"{tag}_{args.split}_samples.jsonl", "w", encoding="utf-8") as f:
                for pr, rf in zip(preds, refs):
                    f.write(json.dumps({"pred": pr, "ref": rf}) + "\n")

    with open(OUT_DIR / f"{tag}_{args.split}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
