import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from typing import Callable


class ContentStyleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        content_path: str = "data/coco/",
        style_path: str = "data/wikiart/",
        resize_size: int = 512,
        crop_size: int = 256,
        batch_size: int = 8,
        workers: int = 4,
    ):
        """Content and style image data module

        Args:
            content_path: Path to content images directory
            style_path: Path to style images directory
            resize_size: Size of resize transformation
            crop_size: Size of random crop transformation
            batch_size: Number of batch samples
            workers: Number of data loader workers
        """
        super().__init__()
        self.content_path = content_path
        self.style_path = style_path
        self.batch_size = batch_size
        self.workers = workers

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            # Load all content and style images
            content_all = SimpleDataset(self.content_path, self.transforms)
            style_all = SimpleDataset(self.style_path, self.transforms)

            # Split into train/val
            self.content_train, self.content_val = data.random_split(
                content_all, [len(content_all) - 50, 50]
            )
            self.style_train, self.style_val = data.random_split(
                style_all, [len(style_all) - 50, 50]
            )

    def train_dataloader(self):
        return {
            "content": DataLoader(
                self.content_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
            ),
            "style": DataLoader(
                self.style_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.workers,
                pin_memory=True,
            ),
        }

    def val_dataloader(self):
        loaders = {
            "content": DataLoader(
                self.content_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
            ),
            "style": DataLoader(
                self.style_val,
                batch_size=1,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
            ),
        }

        return CombinedLoader(loaders, "max_size_cycle")


class SimpleDataset(data.Dataset):
    def __init__(self, root: str, transforms: Callable):
        """Image dataset from directory

        Args:
            root: Path to directory
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        self.paths = os.listdir(root)
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.paths[index])).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)
