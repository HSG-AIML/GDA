import functools
from torchgeo.datasets import LEVIRCDPlus
from torchgeo.datamodules.utils import dataset_split
from torchgeo.transforms.transforms import _RandomNCrop
import lightning.pytorch as pl

import torch
from torch.utils.data import DataLoader
import kornia.augmentation as K

from torchvision.transforms import Compose


class LEVIRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=32,
        num_workers=0,
        val_split_pct=0.2,
        patch_size=(256, 256),
        return_one_image=False,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split_pct = val_split_pct
        self.patch_size = patch_size
        self.return_one_image = return_one_image
        self.kwargs = kwargs

    def on_after_batch_transfer(self, batch, batch_idx):
        if (
            hasattr(self, "trainer")
            and self.trainer is not None
            and hasattr(self.trainer, "training")
            and self.trainer.training
        ):
            # Kornia expects masks to be floats with a channel dimension
            x = batch["image"]
            y = batch["mask"].float().unsqueeze(1)

            train_augmentations = K.AugmentationSequential(
                K.RandomRotation(p=0.5, degrees=90),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomCrop(self.patch_size),
                K.RandomSharpness(p=0.5),
                data_keys=["input", "mask"],
            )
            x, y = train_augmentations(x, y)

            # torchmetrics expects masks to be longs without a channel dimension
            batch["image"] = x
            batch["mask"] = y.squeeze(1).long()

        return batch

    def preprocess(self, sample, only_change=False):
        sample["image"] = (sample["image"] / 255.0).float()
        sample["image"] = torch.flatten(sample["image"], 0, 1)
        sample["mask"] = sample["mask"].long()

        loc = torch.randint(
            0,
            min(sample["image"].shape[-1], sample["image"].shape[-2]) - 224,
            size=(1, 2),
        )
        mask = torch.zeros(sample["mask"].shape)

        if only_change and sample["mask"].sum() >= 1000:
            # only sample image pairs with change
            counter = 0
            while mask.sum() < 100:
                x, y = torch.randint(
                    0,
                    min(sample["image"].shape[-1], sample["image"].shape[-2]) - 224,
                    size=(2, 1),
                )
                patch = sample["image"][:, x : x + 224, y : y + 224]
                mask = sample["mask"][x : x + 224, y : y + 224]
                counter += 1
                if counter >= 10:
                    break

        else:
            x, y = torch.randint(
                0,
                min(sample["image"].shape[-1], sample["image"].shape[-2]) - 224,
                size=(2, 1),
            )
            patch = sample["image"][:, x : x + 224, y : y + 224]
            mask = sample["mask"][x : x + 224, y : y + 224]

        sample["image"] = patch
        sample["mask"] = mask

        if self.return_one_image:
            if torch.rand(1) > 0.5:
                sample["image"] = sample["image"][:3]
            else:
                sample["image"] = sample["image"][3:]
        return sample

    def prepare_data(self):
        LEVIRCDPlus(split="train", **self.kwargs)

    def setup(self, stage=None):
        train_transforms = Compose(
            [functools.partial(self.preprocess, only_change=True)]
        )
        test_transforms = Compose([self.preprocess])

        train_dataset = LEVIRCDPlus(
            split="train", transforms=train_transforms, **self.kwargs
        )
        train_dataset.classes = ["change", "no change"]

        if self.val_split_pct > 0.0:
            self.train_dataset, self.val_dataset, _ = dataset_split(
                train_dataset, val_pct=self.val_split_pct, test_pct=0.0
            )
            self.train_dataset.classes = train_dataset.classes
            self.val_dataset.classes = train_dataset.classes
        else:
            self.train_dataset = train_dataset
            self.val_dataset = train_dataset

        self.test_dataset = LEVIRCDPlus(
            split="test", transforms=test_transforms, **self.kwargs
        )
        self.test_dataset.classes = ["change", "no change"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
