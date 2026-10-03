"""
CLIP-style image-text alignment of the MIM ViT on SkyScript captions.

    python scripts/clip_align.py --name main --epochs 15
    python scripts/clip_align.py --name smoke --max-steps 30 --limit 2560     # smoke test

Image tower: MIM ViT -> mean of patch tokens -> Linear.
Text tower : a separate copy of the pretrained LM, hidden state at the EOS
             position -> Linear. (The original LM is untouched, so the VLM
             can still start from lm_final.pt.)
Loss       : symmetric InfoNCE with a learnable temperature. Pairs whose
             captions are identical are masked out of the negatives (templated
             captions repeat a lot, so in-batch false negatives are common).

The aligned ViT is saved in the same format as the MIM checkpoints
("model_state_dict" with a "vit." prefix), so benchmark_probe.py, linear_probe.py
and train_vlm.py (--vit-ckpt) load it unchanged.
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vlm import (
    IMAGE_SIZE,
    LM_CHECKPOINT,
    MIM_CHECKPOINT,
    TOKENIZER_PATH,
    CaptionDataset,
    load_lm,
    load_vit,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT = ROOT / "checkpoints" / "clip"
MAX_TEXT_TOKENS = 76
SEED = 42


# ============================================================
# Model
# ============================================================

class ClipModel(nn.Module):

    def __init__(self, vit, lm, embed_dim=512):
        super().__init__()

        self.vit = vit
        self.lm = lm

        self.image_proj = nn.Linear(vit.norm.normalized_shape[0], embed_dim, bias=False)
        self.text_proj = nn.Linear(lm.dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        self.grad_checkpointing = True

    def _run(self, block, x):
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def encode_image(self, images):
        v = self.vit
        x = v.patch_embed(images)
        x = torch.cat([v.cls_token.expand(x.size(0), -1, -1), x], dim=1) + v.pos_embed

        for block in v.blocks:
            x = self._run(block, x)

        x = v.norm(x)[:, 1:].mean(dim=1)               # mean of patch tokens
        return F.normalize(self.image_proj(x).float(), dim=-1)

    def encode_text(self, ids, lengths):
        lm = self.lm
        x = lm.token_embedding(ids)

        for layer in lm.layers:
            x = self._run(layer, x)

        x = lm.norm(x)
        x = x[torch.arange(x.size(0), device=x.device), lengths - 1]   # EOS position
        return F.normalize(self.text_proj(x).float(), dim=-1)


def clip_loss(img, txt, logit_scale, same):
    """Symmetric InfoNCE; `same[i, j]` marks identical captions (i != j) as non-negatives."""

    logits = logit_scale.exp().clamp(max=100) * img @ txt.t()
    logits = logits.masked_fill(same, float("-inf"))

    labels = torch.arange(img.size(0), device=img.device)

    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


# ============================================================
# Data
# ============================================================

class ClipDataset(CaptionDataset):

    def __init__(self, manifest, tokenizer, train, limit=None):
        super().__init__(manifest, tokenizer, train=train, limit=limit)

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.9, 1.1), antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])

    def __getitem__(self, idx):
        image, ids = super().__getitem__(idx)
        # CaptionDataset caps captions at 128 tokens; CLIP uses a shorter cap, EOS kept last
        ids = torch.cat([ids[:-1][: MAX_TEXT_TOKENS - 1], ids[-1:]])
        return image, ids


def collate(batch):
    images = torch.stack([b[0] for b in batch])

    lengths = torch.tensor([len(b[1]) for b in batch])
    ids = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long)

    for i, (_, t) in enumerate(batch):
        ids[i, : len(t)] = t

    keys = torch.tensor([hash(tuple(b[1].tolist())) & 0x7FFFFFFFFFFFFFFF for b in batch])

    return images, ids, lengths, keys


# ============================================================
# Evaluation: image <-> text retrieval (a hit = the retrieved caption text matches)
# ============================================================

@torch.no_grad()
def retrieval(model, loader):
    model.eval()

    imgs, txts, keys = [], [], []

    for images, ids, lengths, k in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            imgs.append(model.encode_image(images.to(DEVICE)))
            txts.append(model.encode_text(ids.to(DEVICE), lengths.to(DEVICE)))
        keys.append(k)

    img, txt, key = torch.cat(imgs), torch.cat(txts), torch.cat(keys).to(DEVICE)
    sim = img @ txt.t()                                  # [N images, N captions]
    match = key[:, None] == key[None, :]                 # caption text identical

    out = {}
    for name, s, m in [("i2t", sim, match), ("t2i", sim.t(), match.t())]:
        order = s.argsort(dim=1, descending=True)
        hits = m.gather(1, order)                        # correct-ness in ranked order
        for k in (1, 5, 10):
            out[f"{name}_R@{k}"] = hits[:, :k].any(dim=1).float().mean().item()

    return out


# ============================================================
# Training
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max-steps", type=int, help="stop early (smoke tests); LR schedule still follows --epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr-vit", type=float, default=1e-4)
    p.add_argument("--lr-text", type=float, default=5e-5)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--val-limit", type=int, default=1000)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-checkpointing", action="store_true")
    return p.parse_args()


def lr_scale(step, warmup, total, floor=0.01):
    if step < warmup:
        return (step + 1) / warmup

    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model, args):
    groups = []

    def add(params, lr):
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        for ps, wd in ((decay, args.weight_decay), (no_decay, 0.0)):
            if ps:
                groups.append({"params": ps, "lr": lr, "base_lr": lr, "weight_decay": wd})

    add(list(model.vit.parameters()), args.lr_vit)
    add(list(model.lm.parameters()), args.lr_text)
    add(list(model.image_proj.parameters()) + list(model.text_proj.parameters()), args.lr_head)
    groups.append(
        {"params": [model.logit_scale], "lr": args.lr_head, "base_lr": args.lr_head, "weight_decay": 0.0}
    )

    return torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-6)


def save_vit_checkpoint(path, model, epoch, step):
    """Aligned ViT in the same format as mim_epoch_*.pt."""

    base = torch.load(MIM_CHECKPOINT, map_location="cpu", weights_only=False)["config"]

    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": {f"vit.{k}": v for k, v in model.vit.state_dict().items()},
            "config": base,
        },
        path,
    )


def save_state(path, model, optimizer, epoch, step, args):
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        tmp,
    )
    os.replace(tmp, path)


def main():
    args = parse_args()

    out_dir = OUT_ROOT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))

    model = ClipModel(load_vit(), load_lm()).to(DEVICE)
    model.grad_checkpointing = not args.no_checkpointing

    optimizer = make_optimizer(model, args)

    train_ds = ClipDataset("train_200k.csv", tokenizer, train=True, limit=args.limit)
    val_ds = ClipDataset("val_5k.csv", tokenizer, train=False, limit=args.val_limit)

    steps_per_epoch = len(train_ds) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    stop_at = min(total_steps, args.max_steps or total_steps)

    val_loader = DataLoader(
        val_ds, batch_size=128, num_workers=4, collate_fn=collate
    )

    step = epoch = 0
    latest = out_dir / "clip_latest.pt"

    if args.resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        step, epoch = state["step"], state["epoch"]
        print(f"Resumed from epoch {epoch}, step {step}")

    total = sum(p.numel() for p in model.parameters())
    print(
        f"Parameters: {total / 1e6:.1f}M | {len(train_ds):,} pairs | {steps_per_epoch} steps/epoch | "
        f"{total_steps} total steps | batch {args.batch_size}",
        flush=True,
    )

    log_path = out_dir / "train_log.csv"
    log_f = open(log_path, "a" if args.resume and log_path.exists() else "w", newline="")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["epoch", "step", "train_loss", "temperature", "i2t_R@1", "i2t_R@5", "i2t_R@10",
                      "t2i_R@1", "t2i_R@5", "t2i_R@10"])

    start = time.time()
    start_step = step

    while step < stop_at:

        g = torch.Generator()
        g.manual_seed(SEED + epoch)
        loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=False,
            collate_fn=collate, generator=g, persistent_workers=False,
        )

        skip = step - epoch * steps_per_epoch
        running, n = 0.0, 0

        for i, (images, ids, lengths, keys) in enumerate(loader):

            if i < skip:
                continue

            if step >= stop_at:
                break

            scale = lr_scale(step, args.warmup_steps, total_steps)
            for grp in optimizer.param_groups:
                grp["lr"] = grp["base_lr"] * scale

            model.train()
            images = images.to(DEVICE, non_blocking=True)
            ids = ids.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE, non_blocking=True)
            keys = keys.to(DEVICE, non_blocking=True)

            same = (keys[:, None] == keys[None, :]) & ~torch.eye(len(keys), dtype=torch.bool, device=DEVICE)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                loss = clip_loss(
                    model.encode_image(images), model.encode_text(ids, lengths),
                    model.logit_scale, same,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            running += loss.item()
            n += 1

            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{total_steps} | loss {running / n:.4f} | "
                    f"T {1 / model.logit_scale.exp().item():.3f} | "
                    f"{(step - start_step) / (time.time() - start):.2f} step/s | "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB",
                    flush=True,
                )

        # Epoch finished (or stopped early): evaluate, log, checkpoint
        finished_epoch = step >= (epoch + 1) * steps_per_epoch
        if finished_epoch or step >= stop_at:
            r = retrieval(model, val_loader)
            print(
                f"  [epoch {epoch}] val retrieval ({len(val_ds)}): "
                + " ".join(f"{k} {v:.3f}" for k, v in r.items()),
                flush=True,
            )
            log.writerow([epoch, step, f"{running / max(n, 1):.4f}",
                          f"{1 / model.logit_scale.exp().item():.4f}"]
                         + [f"{r[k]:.4f}" for k in ("i2t_R@1", "i2t_R@5", "i2t_R@10",
                                                    "t2i_R@1", "t2i_R@5", "t2i_R@10")])
            log_f.flush()

            if finished_epoch:
                epoch += 1
                save_state(latest, model, optimizer, epoch, step, args)
                save_vit_checkpoint(out_dir / "vit_latest.pt", model, epoch, step)

                if epoch % 5 == 0:
                    save_vit_checkpoint(out_dir / f"vit_epoch_{epoch:03d}.pt", model, epoch, step)

    save_vit_checkpoint(out_dir / "vit_final.pt", model, epoch, step)
    log_f.close()
    print(f"Done. Aligned ViT: {out_dir / 'vit_final.pt'}")


if __name__ == "__main__":
    main()
