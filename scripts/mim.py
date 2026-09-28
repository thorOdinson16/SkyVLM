import torch
import torch.nn as nn


class MIMModel(nn.Module):
    def __init__(
        self,
        vit,
        patch_size=16,
        embed_dim=512,
        in_channels=3,
        norm_pix_loss=False,
    ):
        super().__init__()

        self.vit = vit
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.norm_pix_loss = norm_pix_loss

        self.patch_dim = (
            patch_size * patch_size * in_channels
        )

        # Learned embedding that stands in for hidden patches
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Reconstruction head
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.patch_dim),
        )

    def patchify(self, images):
        """
        Convert images:

            [B, C, H, W]

        into patches:

            [B, N, C * P * P]
        """

        B, C, H, W = images.shape
        p = self.patch_size

        assert H % p == 0
        assert W % p == 0

        h = H // p
        w = W // p

        x = images.reshape(
            B,
            C,
            h,
            p,
            w,
            p,
        )

        x = x.permute(
            0,
            2,
            4,
            1,
            3,
            5,
        )

        x = x.reshape(
            B,
            h * w,
            C * p * p,
        )

        return x

    def unpatchify(
        self,
        patches,
        height,
        width,
    ):
        """
        Convert:

            [B, N, C * P * P]

        back to:

            [B, C, H, W]
        """

        B, N, D = patches.shape

        p = self.patch_size
        C = self.in_channels

        h = height // p
        w = width // p

        assert N == h * w
        assert D == C * p * p

        x = patches.reshape(
            B,
            h,
            w,
            C,
            p,
            p,
        )

        x = x.permute(
            0,
            3,
            1,
            4,
            2,
            5,
        )

        return x.reshape(
            B,
            C,
            height,
            width,
        )

    def random_mask(
        self,
        batch_size,
        num_patches,
        mask_ratio,
        device,
    ):
        """
        Generate random boolean mask.

        True  = masked
        False = visible
        """

        num_masked = int(
            num_patches * mask_ratio
        )

        noise = torch.rand(
            batch_size,
            num_patches,
            device=device,
        )

        ids = torch.argsort(
            noise,
            dim=1,
        )

        mask = torch.zeros(
            batch_size,
            num_patches,
            dtype=torch.bool,
            device=device,
        )

        mask.scatter_(
            1,
            ids[:, :num_masked],
            True,
        )

        return mask

    def forward(
        self,
        images,
        mask_ratio=0.5,
    ):
        # Original image patches = reconstruction targets
        targets = self.patchify(images)

        if self.norm_pix_loss:
            # MAE: predict per-patch normalized pixels
            mean = targets.mean(dim=-1, keepdim=True)
            var = targets.var(dim=-1, keepdim=True)
            targets = (targets - mean) / (var + 1e-6) ** 0.5

        batch_size = images.size(0)
        num_patches = targets.size(1)

        # Generate mask
        mask = self.random_mask(
            batch_size,
            num_patches,
            mask_ratio,
            images.device,
        )

        # Encode, with masked patches replaced by the mask token
        features = self.vit(
            images,
            mask=mask,
            mask_token=self.mask_token,
        )

        # Ignore CLS token
        patch_features = features[:, 1:, :]

        # Reconstruct every patch
        predictions = self.decoder(
            patch_features
        )

        return (
            predictions,
            targets,
            mask,
        )


def mim_loss(
    predictions,
    targets,
    mask,
):
    """
    Calculate reconstruction loss only
    over masked patches.
    """

    loss_per_patch = (
        (predictions - targets) ** 2
    ).mean(dim=-1)

    masked_loss = loss_per_patch[mask]

    return masked_loss.mean()