import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import torch
import lightning.pytorch as pl

from src.datasets import BENGE


S2_MEAN = torch.tensor(
    [
        1354.40546513,
        1118.24399958,
        1042.92983953,
        947.62620298,
        1199.47283961,
        1999.79090914,
        2369.22292565,
        2296.82608323,
        732.08340178,
        12.11327804,
        1819.01027855,
        1118.92391149,
        2594.14080798,
    ]
)

S2_STD = torch.tensor(
    [
        245.71762908,
        333.00778264,
        395.09249139,
        593.75055589,
        566.4170017,
        861.18399006,
        1086.63139075,
        1117.98170791,
        404.91978886,
        4.77584468,
        1002.58768311,
        761.30323499,
        1231.58581042,
    ]
)

# after removing negatives in __getitem__
S1_MEAN = torch.tensor([0.25193255, 0.50496706, 0.25193255])
S1_STD = torch.tensor([0.1677641, 0.18358294, 0.1677641])


class BENGEDataModule(pl.LightningDataModule):
    """Pytorch Lightning data module class for ben-ge."""

    s2_mean = S2_MEAN
    s2_std = S2_STD
    s1_mean = S1_MEAN
    s1_std = S1_STD

    def __init__(
        self,
        root="data/",
        batch_size=32,
        num_workers=0,
        s1=True,
        s2=False,
        lulc=True,
        dem=None,
        transforms=None,
        few_shot_k=None,
        few_shot_seed=None,
    ):
        """BENGEDataModule constructor."""
        super(BENGEDataModule).__init__()
        self.root = root
        self.train_batch_size = batch_size
        self.eval_batch_size = batch_size
        self.num_workers = num_workers
        self.s1 = s1
        self.s2 = s2
        self.lulc = lulc
        self.dem = dem
        self.transforms = transforms
        self.few_shot_k = few_shot_k
        self.few_shot_seed = few_shot_seed

    def prepare_data(self):
        """Method to prepare data."""
        pass

    def setup(
        self,
        stage="fit",
        drop_last=False,
    ):
        """Method to setup dataset and corresponding splits."""
        for split in ["train", "val", "test"]:
            if self.s1:
                setattr(
                    self,
                    f"{split}_dataset",
                    BENGE(
                        self.root,
                        s2_bands={},
                        lulc=self.lulc,
                        dem=self.dem,
                        split=split,
                        transforms=self.transforms,
                        few_shot_seed=self.few_shot_seed,
                        few_shot_k=self.few_shot_k,
                    ),
                )
            elif self.s2:
                setattr(
                    self,
                    f"dataset_{split}",
                    BENGE(
                        self.root,
                        s1_bands={},
                        lulc=self.lulc,
                        dem=self.dem,
                        split=split,
                        transforms=self.transforms,
                        few_shot_seed=self.few_shot_seed,
                        few_shot_k=self.few_shot_k,
                    ),
                )
            else:
                assert self.dem
                setattr(
                    self,
                    f"dataset_{split}",
                    BENGE(
                        self.root,
                        s1_bands={},
                        s2_bands={},
                        dem=True,
                        split=split,
                        transforms=self.transforms,
                        few_shot_seed=self.few_shot_seed,
                        few_shot_k=self.few_shot_k,
                    ),
                )

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
