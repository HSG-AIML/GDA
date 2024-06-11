import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import torch
import lightning.pytorch as pl

from src.datasets import Caltech256

# ImageNet statistics
# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]

# Caltech256 training set
MEAN = torch.tensor([0.5407, 0.5139, 0.4840])
STD = torch.tensor([0.3071, 0.3028, 0.3145])


class Caltech256DataModule(pl.LightningDataModule):
    mean = MEAN
    std = STD

    def __init__(self, root="data/", batch_size=32, num_workers=0, transforms=None):
        super(Caltech256DataModule).__init__()
        self.root = root
        self.train_batch_size = batch_size
        self.eval_batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms

    def setup(
        self,
        stage="fit",
        drop_last=False,
    ):
        """Method to setup dataset and corresponding splits."""
        for split in ["train", "val", "test"]:
            ds = Caltech256(self.root, split=split, transforms=self.transforms)
            setattr(self, f"{split}_dataset", ds)

        self.drop_last = drop_last

    def train_dataloader(self):
        """Return training dataset loader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        """Return validation dataset loader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        """Return test dataset loader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )
