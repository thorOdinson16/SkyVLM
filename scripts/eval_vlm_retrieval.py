"""
Image <-> text retrieval with the VLM itself: rank captions by log p(caption | image).

    python scripts/eval_vlm_retrieval.py --ckpt checkpoints/vlm/stage2/vlm_final.pt --split val --n 1000

Same protocol as eval_retrieval.py (first N pairs, N candidates each way, a hit is a
retrieved caption with the same text as the ground truth), so the numbers sit next to
the CLIP towers and SkyCLIP.

Score matrix S[i, c] = log p(caption c | image i), summed over caption tokens + EOS.
  t2i  : each caption ranks the images by S[:, c]            (the caption is fixed, so S is the right score)
  i2t  : each image ranks the captions. Raw S favours short captions, so three variants:
         sum   S[i, c]
         mean  S[i, c] / n_tokens(c)
         pmi   S[i, c] - log p(c), with log p(c) = logsumexp_i S[i, c] - log N  (marginal over the N images)

Speed: the image prefix is bidirectional and does not depend on the caption, so its
keys/values are computed once per image and reused for every candidate caption.
--check compares the cached scorer with the model's ordinary forward pass.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from lm import rotate_half
from vlm import IMAGE_DIR, IMAGE_SIZE, MAX_CAPTION_TOKENS, TOKENIZER_PATH, build_vlm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = ROOT / "checkpoints" / "eval" / "retrieval"


def load_pairs(split, n):
    name = {"val": "val_5k.csv", "test": "test_30k.csv"}[split]
    df = pd.read_csv(ROOT / "dataset" / "manifests" / name, keep_default_na=False)
    df = df[df["caption"].str.strip() != ""].reset_index(drop=True).iloc[:n]
    return df["image_path"].tolist(), df["caption"].str.strip().tolist()


def rope(x, cos, sin, start):
    T = x.shape[-2]
    c = cos[start:start + T][None, None]
    s = sin[start:start + T][None, None]
    return x * c + rotate_half(x) * s


def layer_forward(layer, x, start, past=None):
    """One transformer block. past=None: bidirectional prefix pass, returns (x, (k, v)).
    past=(k, v) [1,H,P,hd]: x are caption tokens (causal among themselves, full view of the prefix)."""

    a = layer.attention
    B, T, C = x.shape
    H, hd = a.num_heads, a.head_dim

    h = layer.norm1(x)
    q = a.q_proj(h).view(B, T, H, hd).transpose(1, 2)
    k = a.k_proj(h).view(B, T, H, hd).transpose(1, 2)
    v = a.v_proj(h).view(B, T, H, hd).transpose(1, 2)
    q, k = rope(q, a.rope_cos, a.rope_sin, start), rope(k, a.rope_cos, a.rope_sin, start)

    if past is None:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        kv = (k, v)
    else:
        pk, pv = past
        P = pk.size(2)
        mask = torch.cat([
            torch.ones(T, P, dtype=torch.bool, device=x.device),
            torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)),
        ], dim=1)
        k_all = torch.cat([pk.expand(B, -1, -1, -1), k], dim=2)
        v_all = torch.cat([pv.expand(B, -1, -1, -1), v], dim=2)
        y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        kv = None

    x = x + a.o_proj(y.transpose(1, 2).reshape(B, T, C))
    x = x + layer.ffn(layer.norm2(x))
    return x, kv


@torch.no_grad()
def encode_prefix(model, image):
    """image [1,3,H,W] -> (per-layer prefix K/V, log-probs of the first caption token [V])."""

    x = model.image_prefix(image)
    P = x.size(1)
    cache = []
    for layer in model.lm.layers:
        x, kv = layer_forward(layer, x, 0)
        cache.append(kv)
    h = model.lm.norm(x[:, -1])
    return cache, model.lm.lm_head(h).float().log_softmax(-1)[0], P


@torch.no_grad()
def score_batch(model, cache, first_logp, P, tokens, targets):
    """tokens [C,n] caption inputs (ids[:-1], right-padded), targets [C,n+1] (ids, pad = -100).
    Returns summed log p(caption | image) per row [C]."""

    lm = model.lm
    C = tokens.size(0)

    x = lm.token_embedding(tokens)
    for layer, kv in zip(lm.layers, cache):
        x, _ = layer_forward(layer, x, P, past=kv)

    logp = lm.lm_head(lm.norm(x)).float().log_softmax(-1)               # [C, n, V]
    valid = targets[:, 1:] >= 0
    tgt = targets[:, 1:].clamp(min=0)
    rest = logp.gather(-1, tgt[..., None])[..., 0] * valid               # tokens 1..n
    first = first_logp[targets[:, 0]]                                    # token 0

    return first + rest.sum(1)


def tokenize(captions, tok):
    seqs = [tok.encode(c)[:MAX_CAPTION_TOKENS] + [tok.eos_id()] for c in captions]
    return seqs


def make_batches(seqs, max_tokens=6000):
    """Length-sorted batches of (indices, tokens [C,n], targets [C,n+1]) kept on the GPU."""

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * len(seqs[i]) > max_tokens:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)

    out = []
    for b in batches:
        n = max(len(seqs[i]) for i in b)
        tokens = torch.zeros(len(b), n - 1, dtype=torch.long)
        targets = torch.full((len(b), n), -100, dtype=torch.long)
        for r, i in enumerate(b):
            s = torch.tensor(seqs[i])
            tokens[r, : len(s) - 1] = s[:-1]
            targets[r, : len(s)] = s
        out.append((torch.tensor(b, device=DEVICE), tokens.to(DEVICE), targets.to(DEVICE)))
    return out


def rank_metrics(sim, match, prefix):
    """sim [N,N] (rows = queries). A hit is any top-k candidate that is a match."""

    order = sim.argsort(dim=1, descending=True)
    hits = match.gather(1, order)
    return {f"{prefix}_R@{k}": round(hits[:, :k].any(dim=1).float().mean().item(), 4) for k in (1, 5, 10)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--max-batch-tokens", type=int, default=6000)
    p.add_argument("--check", action="store_true", help="compare with the ordinary forward pass and exit")
    p.add_argument("--tag")
    args = p.parse_args()

    tok = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    model = build_vlm()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.to(DEVICE).eval()

    paths, captions = load_pairs(args.split, args.n)
    N = len(paths)
    seqs = tokenize(captions, tok)
    batches = make_batches(seqs, args.max_batch_tokens)

    tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])

    def load_image(i):
        return tf(Image.open(IMAGE_DIR / paths[i]).convert("RGB"))[None].to(DEVICE)

    if args.check:
        # cached scorer vs. the model's own forward on a few (image, caption) pairs
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            img = load_image(0)
            cache, first, P = encode_prefix(model, img)
            for idx, tokens, targets in batches[:2]:
                got = score_batch(model, cache, first, P, tokens, targets)
                for r in range(min(3, len(idx))):
                    i = int(idx[r])
                    n = int((targets[r] >= 0).sum())
                    _, loss = model(img, tokens[r:r + 1, : n - 1], targets[r:r + 1, :n])
                    print(f"caption {i}: cached {got[r].item():.3f}  reference {-loss.item() * n:.3f}", flush=True)
        return

    S = torch.zeros(N, N, device=DEVICE)                                  # S[image, caption]
    start = time.time()

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        for i in range(N):
            cache, first, P = encode_prefix(model, load_image(i))
            for idx, tokens, targets in batches:
                S[i, idx] = score_batch(model, cache, first, P, tokens, targets)

            if (i + 1) % 50 == 0:
                print(f"image {i + 1}/{N}  {time.time() - start:.0f}s", flush=True)

    ids = {}
    key = torch.tensor([ids.setdefault(c, len(ids)) for c in captions], device=DEVICE)
    match = key[:, None] == key[None, :]
    n_tok = torch.tensor([len(s) for s in seqs], device=DEVICE, dtype=torch.float)

    log_marg = torch.logsumexp(S, dim=0) - math.log(N)                    # log p(c) over the N images

    result = {"n": N, "unique_captions": len(ids), "split": args.split, "ckpt": args.ckpt, "step": state.get("step")}
    result.update(rank_metrics(S.t(), match.t(), "t2i"))                       # caption -> images
    result.update({f"{k}_sum": v for k, v in rank_metrics(S, match, "i2t").items()})
    result.update({f"{k}_mean": v for k, v in rank_metrics(S / n_tok[None], match, "i2t").items()})
    result.update({f"{k}_pmi": v for k, v in rank_metrics(S - log_marg[None], match, "i2t").items()})

    print(json.dumps(result), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"vlm_{Path(args.ckpt).parent.name}_{args.split}_{N}"
    with open(OUT_DIR / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    torch.save(S.cpu(), OUT_DIR / f"{tag}_scores.pt")


if __name__ == "__main__":
    main()
