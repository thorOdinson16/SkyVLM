"""
SkyVLM training: MIM ViT + MLP projector + pretrained LM on image-caption pairs.

    # Stage 1: projector (+ boundary tokens) only, ViT and LM frozen
    python scripts/train_vlm.py --name stage1 --max-steps 3000 --lr-projector 1e-3

    # Stage 2: also unfreeze the LM (and optionally the ViT) at lower LRs
    python scripts/train_vlm.py --name stage2 --init checkpoints/vlm/stage1/vlm_final.pt \\
        --train-lm --lr-lm 1e-4 --lr-projector 3e-4 --max-steps 20000

    python scripts/train_vlm.py --name smoke --max-steps 20 --limit 640   # smoke test

Every eval logs val loss with the real images AND with images shuffled across
the batch; the gap between them is the evidence that the model uses the image.
Keep --max-steps / --batch-size / --warmup-steps identical when resuming.
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from vlm import TOKENIZER_PATH, CaptionDataset, build_vlm, collate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--init", help="VLM checkpoint to start from (e.g. stage 1)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lm-ckpt", help="override LM checkpoint")
    p.add_argument("--vit-ckpt", help="override ViT checkpoint (e.g. a CLIP-aligned ViT)")
    p.add_argument("--random-vit", action="store_true", help="baseline: randomly initialized ViT")
    p.add_argument("--train-lm", action="store_true")
    p.add_argument("--train-vit", action="store_true")
    p.add_argument("--lr-projector", type=float, default=1e-3)
    p.add_argument("--lr-lm", type=float, default=1e-4)
    p.add_argument("--lr-vit", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32, help="micro-batch size")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--keep-every", type=int, default=0, help="also keep a vlm_step_N.pt snapshot every N steps")
    p.add_argument("--val-limit", type=int, default=1000)
    p.add_argument("--limit", type=int, help="use only the first N train pairs")
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def lr_scale(step, warmup, total, floor=0.1):
    if step < warmup:
        return (step + 1) / warmup

    progress = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def set_trainable(model, train_lm, train_vit):
    for p in model.parameters():
        p.requires_grad = False

    new_modules = [model.projector]
    for m in new_modules:
        for p in m.parameters():
            p.requires_grad = True

    model.img_start.requires_grad = True
    model.img_end.requires_grad = True

    if train_lm:
        for p in model.lm.parameters():
            p.requires_grad = True

    if train_vit:
        for p in model.vit.parameters():
            p.requires_grad = True


def make_optimizer(model, args):
    groups = []

    def add(params, lr, decay):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": lr,
                    "base_lr": lr,
                    "weight_decay": args.weight_decay if decay else 0.0,
                }
            )

    def split(module):
        decay = [p for p in module.parameters() if p.ndim >= 2]
        no_decay = [p for p in module.parameters() if p.ndim < 2]
        return decay, no_decay

    for module, lr in [
        (model.projector, args.lr_projector),
        (model.lm, args.lr_lm),
        (model.vit, args.lr_vit),
    ]:
        d, n = split(module)
        add(d, lr, True)
        add(n, lr, False)

    add([model.img_start, model.img_end], args.lr_projector, False)

    return torch.optim.AdamW(groups, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    real = shuffled = 0.0
    n = 0

    for images, text_in, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        text_in = text_in.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            real += model(images, text_in, targets)[1].item()
            shuffled += model(images.roll(1, dims=0), text_in, targets)[1].item()

        n += 1

    return real / n, shuffled / n


def save_checkpoint(path, model, optimizer, step, args, val_real, val_shuffled):
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "val_loss": val_real,
        "val_loss_shuffled": val_shuffled,
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()

    out_dir = ROOT / "checkpoints" / "vlm" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))

    model = build_vlm(
        random_vit=args.random_vit,
        **({"lm_ckpt": args.lm_ckpt} if args.lm_ckpt else {}),
        **({"vit_ckpt": args.vit_ckpt} if args.vit_ckpt else {}),
    )

    if args.init:
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        print(f"Initialized from {args.init} (step {state['step']})")

    model.to(DEVICE)
    set_trainable(model, args.train_lm, args.train_vit)

    optimizer = make_optimizer(model, args)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total / 1e6:.1f}M total, {trainable / 1e6:.2f}M trainable")
    print(f"Training: projector=True lm={args.train_lm} vit={args.train_vit}")

    train_ds = CaptionDataset("train_200k.csv", tokenizer, train=True, limit=args.limit)
    val_ds = CaptionDataset("val_5k.csv", tokenizer, limit=args.val_limit)

    def loader(ds, shuffle, seed=0):
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=args.workers,
            persistent_workers=args.workers > 0,
            pin_memory=False,
            collate_fn=collate,
            generator=g,
        )

    val_loader = loader(val_ds, False)

    step = 0
    latest = out_dir / "vlm_latest.pt"

    if args.resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        step = state["step"]
        print(f"Resumed from step {step}")

    steps_per_epoch = len(train_ds) // (args.batch_size * args.grad_accum)

    log_path = out_dir / "train_log.csv"
    new_log = not log_path.exists() or not args.resume
    log_f = open(log_path, "a" if args.resume else "w", newline="")
    log = csv.writer(log_f)

    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "val_loss_shuffled", "lr_projector"])

    window, window_n = 0.0, 0
    start = time.time()
    start_step = step
    val_real = val_shuffled = float("nan")

    while step < args.max_steps:

        epoch = step // steps_per_epoch
        train_loader = loader(train_ds, True, seed=SEED + epoch)
        skip = (step % steps_per_epoch) * args.grad_accum  # resume mid-epoch: skip seen batches
        micro = []
        batches = iter(train_loader)

        for _ in range(skip):
            next(batches)

        while step < args.max_steps:

            micro = []
            for _ in range(args.grad_accum):
                b = next(batches, None)
                if b is None:
                    break
                micro.append(b)

            if len(micro) < args.grad_accum:
                break  # epoch over; outer loop starts the next one

            scale = lr_scale(step, args.warmup_steps, args.max_steps)
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * scale

            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0

            for images, text_in, targets in micro:
                images = images.to(DEVICE, non_blocking=True)
                text_in = text_in.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
                    _, loss = model(images, text_in, targets)

                (loss / args.grad_accum).backward()
                loss_sum += loss.item() / args.grad_accum

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            step += 1
            window += loss_sum
            window_n += 1

            if step % 50 == 0 or step == args.max_steps:
                elapsed = time.time() - start
                print(
                    f"step {step}/{args.max_steps} | loss {window / window_n:.4f} | "
                    f"{(step - start_step) / elapsed:.2f} step/s | "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                    if DEVICE.type == "cuda"
                    else f"step {step}/{args.max_steps} | loss {window / window_n:.4f}",
                    flush=True,
                )

            if step % args.eval_every == 0 or step == args.max_steps:
                val_real, val_shuffled = evaluate(model, val_loader)
                print(
                    f"  val loss {val_real:.4f} | shuffled-image {val_shuffled:.4f} | "
                    f"image gain {val_shuffled - val_real:+.4f}",
                    flush=True,
                )
                log.writerow(
                    [step, f"{window / window_n:.4f}", f"{val_real:.4f}",
                     f"{val_shuffled:.4f}", f"{optimizer.param_groups[0]['lr']:.2e}"]
                )
                log_f.flush()
                window, window_n = 0.0, 0

            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(latest, model, optimizer, step, args, val_real, val_shuffled)

                if args.keep_every and step % args.keep_every == 0:
                    save_checkpoint(
                        out_dir / f"vlm_step_{step:06d}.pt", model, None,
                        step, args, val_real, val_shuffled,
                    )

    save_checkpoint(out_dir / "vlm_final.pt", model, None, step, args, val_real, val_shuffled)
    log_f.close()
    print(f"Done. Final checkpoint: {out_dir / 'vlm_final.pt'}")


if __name__ == "__main__":
    main()
