# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT-SAR datamodule."""

from typing import Any

import torch

from ..datasets import EuroSATSAR
from torchgeo.datamodules.geo import NonGeoDataModule

MEAN = torch.tensor([-11.1992, -18.0610])
STD = torch.tensor([5.7238, 6.0161])


class EuroSATSARDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EuroSAT dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    mean = MEAN
    std = STD

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new EuroSATDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.EuroSAT`.
        """
        super().__init__(EuroSATSAR, batch_size, num_workers, **kwargs)


class EuroSAT100DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EuroSAT100 dataset.

    Intended for tutorials and demonstrations, not for benchmarking.

    .. versionadded:: 0.5
    """

    mean = MEAN
    std = STD

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new EuroSAT100DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.EuroSAT100`.
        """
        super().__init__(EuroSAT100, batch_size, num_workers, **kwargs)
