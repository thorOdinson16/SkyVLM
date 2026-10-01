"""
SkyVLM: MIM-pretrained ViT -> 2-layer MLP projector -> pretrained LM.

Sequence layout fed to the LM:

    [img_start] [196 projected patch tokens] [img_end] caption tokens...

Attention is prefix-LM: the image block (img_start .. img_end) is fully
bidirectional, caption tokens attend to the whole image and causally to
earlier caption tokens. The loss is on caption tokens (and EOS) only.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from lm import SkyVLMForCausalLM
from vit import SmallViT

IMAGE_DIR = ROOT / "dataset" / "images"
MANIFEST_DIR = ROOT / "dataset" / "manifests"
TOKENIZER_PATH = ROOT / "dataset" / "tokenizer" / "skyvlm.model"
MIM_CHECKPOINT = ROOT / "checkpoints" / "mim" / "main" / "mim_epoch_100.pt"
LM_CHECKPOINT = ROOT / "checkpoints" / "lm" / "lm_final.pt"

IMAGE_SIZE = 224
MAX_CAPTION_TOKENS = 128
IGNORE = -100


# ============================================================
# Pretrained component loaders
# ============================================================

def load_vit(path=MIM_CHECKPOINT):
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})

    vit = SmallViT(
        img_size=cfg.get("image_size", 224),
        patch_size=cfg.get("patch_size", 16),
        embed_dim=cfg.get("embed_dim", 512),
        depth=cfg.get("depth", 8),
        num_heads=cfg.get("num_heads", 8),
        mlp_dim=cfg.get("mlp_dim", 2048),
    )

    # train_mim.py saves the full MIMModel; the encoder is under "vit."
    vit.load_state_dict(
        {
            k[len("vit."):]: v
            for k, v in state["model_state_dict"].items()
            if k.startswith("vit.")
        },
        strict=True,
    )

    return vit


def load_lm(path=LM_CHECKPOINT):
    state = torch.load(path, map_location="cpu", weights_only=False)
    lm = SkyVLMForCausalLM(**state["config"])
    lm.load_state_dict(state["model_state_dict"], strict=True)
    return lm


# ============================================================
# Model
# ============================================================

class SkyVLM(nn.Module):

    def __init__(self, vit, lm, projector_hidden=1024):
        super().__init__()

        self.vit = vit
        self.lm = lm

        vit_dim = vit.norm.normalized_shape[0]
        lm_dim = lm.dim

        self.projector = nn.Sequential(
            nn.Linear(vit_dim, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, lm_dim),
        )

        # Learned boundary tokens, in LM embedding space
        self.img_start = nn.Parameter(torch.randn(1, 1, lm_dim) * 0.02)
        self.img_end = nn.Parameter(torch.randn(1, 1, lm_dim) * 0.02)

        self.num_patches = vit.patch_embed.num_patches
        self.prefix_len = self.num_patches + 2

        # Prefix-LM mask (True = may attend): causal, plus a fully
        # bidirectional block over the image prefix.
        T = lm.max_seq_len
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        mask[: self.prefix_len, : self.prefix_len] = True
        self.register_buffer("attn_mask", mask, persistent=False)

    def image_prefix(self, images):
        """images [B,3,H,W] -> LM-space prefix [B, num_patches + 2, D]."""

        feats = self.vit(images)[:, 1:]            # drop CLS, keep patches
        tokens = self.projector(feats)

        B = tokens.size(0)

        return torch.cat(
            [
                self.img_start.expand(B, -1, -1).to(tokens.dtype),
                tokens,
                self.img_end.expand(B, -1, -1).to(tokens.dtype),
            ],
            dim=1,
        )

    def forward(self, images, text_in, targets=None):
        """
        text_in: [B, n] caption tokens fed to the LM (right-padded)
        targets: [B, n + 1] next-token targets, IGNORE where padded.
                 targets[:, i] is predicted from the position of text_in[:, i-1]
                 (targets[:, 0] from img_end).
        Returns (logits over caption positions [B, n+1, V], loss or None).
        """

        prefix = self.image_prefix(images)
        P = prefix.size(1)

        embeds = torch.cat(
            [prefix, self.lm.token_embedding(text_in).to(prefix.dtype)],
            dim=1,
        )

        T = embeds.size(1)

        logits, _ = self.lm(
            None,
            inputs_embeds=embeds,
            attn_mask=self.attn_mask[:T, :T],
        )

        logits = logits[:, P - 1:]

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, images, eos_id, max_new_tokens=64):
        """Greedy decoding (no KV cache; fine for evaluation-sized runs)."""

        self.eval()

        B = images.size(0)
        prefix = self.image_prefix(images)
        P = prefix.size(1)

        out = torch.empty(B, 0, dtype=torch.long, device=images.device)
        done = torch.zeros(B, dtype=torch.bool, device=images.device)

        for _ in range(max_new_tokens):

            embeds = torch.cat(
                [prefix, self.lm.token_embedding(out).to(prefix.dtype)],
                dim=1,
            )

            T = embeds.size(1)

            logits, _ = self.lm(
                None,
                inputs_embeds=embeds,
                attn_mask=self.attn_mask[:T, :T],
                last_only=True,
            )

            nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            out = torch.cat([out, nxt[:, None]], dim=1)
            done |= nxt == eos_id

            if done.all():
                break

        return out


def build_vlm(vit_ckpt=MIM_CHECKPOINT, lm_ckpt=LM_CHECKPOINT):
    return SkyVLM(load_vit(vit_ckpt), load_lm(lm_ckpt))


# ============================================================
# Data
# ============================================================

class CaptionDataset(Dataset):
    """(image, caption) pairs from a manifest; images are [0, 1] tensors,
    matching how the MIM encoder was pretrained and probed."""

    def __init__(self, manifest, tokenizer, train=False, limit=None):
        df = pd.read_csv(MANIFEST_DIR / manifest, keep_default_na=False)
        df = df[df["caption"].str.strip() != ""].reset_index(drop=True)

        if limit:
            df = df.iloc[:limit]

        self.df = df
        self.tokenizer = tokenizer
        self.train = train

        steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

        if train:
            steps += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]

        steps.append(transforms.ToTensor())
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(IMAGE_DIR / row["image_path"]).convert("RGB")
        image = self.transform(image)

        ids = self.tokenizer.encode(row["caption"].strip())[:MAX_CAPTION_TOKENS]
        ids = ids + [self.tokenizer.eos_id()]

        return image, torch.tensor(ids, dtype=torch.long)


def collate(batch):
    images = torch.stack([b[0] for b in batch])

    n = max(len(b[1]) for b in batch)
    text_in = torch.zeros(len(batch), n - 1, dtype=torch.long)
    targets = torch.full((len(batch), n), IGNORE, dtype=torch.long)

    for i, (_, ids) in enumerate(batch):
        text_in[i, : len(ids) - 1] = ids[:-1]
        targets[i, : len(ids)] = ids

    return images, text_in, targets
