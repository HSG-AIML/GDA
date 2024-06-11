# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ETCI 2021 datamodule."""

import os
from typing import Any

import torch
from torch import Tensor

from ..datasets import ETCI2021
from torchgeo.datamodules.geo import NonGeoDataModule


class ETCI2021DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the ETCI2021 dataset.

    Splits the existing train split from the dataset into train/val with 80/20
    proportions, then uses the existing val dataset as the test data.

    note: this returns 6 bands for 'image', 3x the vv and 3x the vh bands
    the mask consists of 2 bands, the normal water and the flood extend

    .. versionadded:: 0.2
    """

    mean = torch.tensor(
        [
            128.02253931,
            128.02253931,
            128.02253931,
            128.11221701,
            128.11221701,
            128.11221701,
        ]
    )
    std = torch.tensor(
        [89.8145088, 89.8145088, 89.8145088, 95.2797861, 95.2797861, 95.2797861]
    )

    folder = "etci2021"

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        train_split_file=None,
        **kwargs: Any
    ) -> None:
        """Initialize a new ETCI2021DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ETCI2021`.
        """
        super().__init__(ETCI2021, batch_size, num_workers, **kwargs)
        self.train_split_file = train_split_file

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit"]:
            if self.train_split_file is None:
                # default train set
                self.train_split_file = "data/etci2021-train.txt"
            self.train_dataset = ETCI2021(
                split="train", split_file=self.train_split_file, **self.kwargs
            )
        if stage in ["fit", "validate", "val"]:
            self.val_dataset = ETCI2021(
                split="val", split_file="data/etci2021-val.txt", **self.kwargs
            )
        if stage in ["predict", "test"]:
            self.test_dataset = ETCI2021(
                split="test", split_file="data/etci2021-test.txt", **self.kwargs
            )

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if not self.trainer.predicting:
                # Evaluate against flood mask, not water mask
                batch["mask"] = (batch["mask"][:, 1] > 0).long()

        return super().on_after_batch_transfer(batch, dataloader_idx)
