# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""DeepGlobe Land Cover Classification Challenge datamodule."""

from typing import Any, Union

import torch
import kornia.augmentation as K

from ..datasets import DeepGlobeLandCover
from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential
from torchgeo.transforms.transforms import _RandomNCrop
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datamodules.utils import dataset_split


MEAN = torch.tensor([103.7401, 96.6592, 71.8326])
STD = torch.tensor([32.5282, 25.9115, 22.8590])


class DeepGlobeLandCoverDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the DeepGlobe Land Cover dataset.

    Uses the train/test splits from the dataset.
    """

    mean = MEAN
    std = STD

    def __init__(
        self,
        batch_size: int = 64,
        patch_size=64,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new DeepGlobeLandCoverDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.DeepGlobeLandCover`.
        """
        super().__init__(DeepGlobeLandCover, 1, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(self.patch_size, batch_size),
            data_keys=["image", "mask"],
        )

    def setup(self, stage) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        # only train data has labels..
        self.dataset = DeepGlobeLandCover(split="train", **self.kwargs)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            self.dataset, self.val_split_pct, self.test_split_pct
        )
        self.train_dataset.classes = self.dataset.classes
        self.val_dataset.classes = self.dataset.classes
        self.test_dataset.classes = self.dataset.classes
