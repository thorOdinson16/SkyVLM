import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset" / "manifests" / "train_200k.csv"


class SkyScriptDataset(Dataset):
    def __init__(self, manifest, image_size=224, augment=False):
        self.df = pd.read_csv(manifest)

        if augment:
            # MAE-style pretraining augmentation
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.2, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = ROOT / "dataset" / "images" / row["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "caption": row["caption"],
            "image_path": row["image_path"],
        }


def main():
    print("Manifest:", MANIFEST)
    print("Exists:", MANIFEST.exists())

    dataset = SkyScriptDataset(MANIFEST)

    print("Dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    images = batch["image"]

    print("\nBatch information")
    print("-----------------")
    print("Image tensor shape:", images.shape)
    print("Image dtype:", images.dtype)
    print("Image min:", images.min().item())
    print("Image max:", images.max().item())
    print("Captions:", len(batch["caption"]))

    print("\nSample paths:")
    for path in batch["image_path"]:
        print(" ", path)

    print("\nSample captions:")
    for caption in batch["caption"]:
        print(" ", caption)

    if torch.cuda.is_available():
        images = images.cuda()

        print("\nGPU test")
        print("--------")
        print("GPU:", torch.cuda.get_device_name(0))
        print("Tensor device:", images.device)
        print("Tensor shape:", images.shape)


if __name__ == "__main__":
    main()