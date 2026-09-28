import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=512):
        super().__init__()

        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)                 # B, D, 14, 14
        x = x.flatten(2)                 # B, D, 196
        x = x.transpose(1, 2)            # B, 196, D
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        mlp_dim=2048,
        dropout=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)

        x, _ = self.attn(
            x, x, x,
            need_weights=False,
        )

        x = residual + x

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class SmallViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_dim=2048,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, mask=None, mask_token=None):
        x = self.patch_embed(x)

        # MIM: swap masked patch embeddings for the learned mask
        # token before positions are added, so the encoder knows
        # both that a patch is hidden and where it is.
        if mask is not None:
            x = torch.where(
                mask.unsqueeze(-1),
                mask_token.to(x.dtype),
                x,
            )

        batch_size = x.size(0)

        cls = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SmallViT().to(device)

    # Dummy batch
    x = torch.randn(
        4,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        output = model(x)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", total_params)
    print("Parameters (M):", round(total_params / 1e6, 2))

    print("Input shape: ", tuple(x.shape))
    print("Output shape:", tuple(output.shape))

    print("Patch tokens:", model.patch_embed.num_patches)
    print("Embedding dim:", output.shape[-1])

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2

        print("GPU memory allocated (MB):", round(allocated, 2))
        print("GPU memory reserved (MB):", round(reserved, 2))


if __name__ == "__main__":
    main()