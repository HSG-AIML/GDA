# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT SAR dataset."""

import os
from collections.abc import Sequence
from typing import Callable, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor

from torchgeo.datasets.geo import NonGeoClassificationDataset
from torchgeo.datasets.utils import (
    check_integrity,
    download_url,
    extract_archive,
    rasterio_loader,
)


class EuroSATSAR(NonGeoClassificationDataset):
    """EuroSAT SAR dataset.

    Dataset format:

    * rasters are 2-channel GeoTiffs
    * labels are values in the range [0,9]

    Dataset classes:

    * Industrial Buildings
    * Residential Buildings
    * Annual Crop
    * Permanent Crop
    * River
    * Sea and Lake
    * Herbaceous Vegetation
    * Highway
    * Pasture
    * Forest

    """

    url = "https://huggingface.co/datasets/wangyi111/EuroSAT-SAR/resolve/main/EuroSAT-SAR.zip"
    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/eurosat-train.txt",  # noqa: E501
        "val": "https://storage.googleapis.com/remote_sensing_representations/eurosat-val.txt",  # noqa: E501
        "test": "https://storage.googleapis.com/remote_sensing_representations/eurosat-test.txt",  # noqa: E501
    }
    split_md5s = {
        "train": "908f142e73d6acdf3f482c5e80d851b1",
        "val": "95de90f2aa998f70a3b2416bfe0687b4",
        "test": "7ae5ab94471417b6e315763121e67c5f",
    }
    classes = [
        "Industrial Buildings",
        "Residential Buildings",
        "Annual Crop",
        "Permanent Crop",
        "River",
        "Sea and Lake",
        "Herbaceous Vegetation",
        "Highway",
        "Pasture",
        "Forest",
    ]

    all_band_names = (
        "VV",
        "VH",
    )

    rgb_bands = ("VV", "VH", "VV")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: Sequence[str] = BAND_SETS["all"],
        transforms=None,
        download: bool = False,
        checksum: bool = False,
        train_split_file_suffix=".txt",
    ) -> None:
        """Initialize a new EuroSAT-SAR dataset instance.

        Args:
            root: root directory where dataset can be found
            split: one of "train", "val", or "test"
            bands: a sequence of band names to load
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if ``download=False`` and data is not found, or checksums
                don't match

        .. versionadded:: 0.3
           The *bands* parameter.
        """
        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum

        assert split in ["train", "val", "test"]

        self._validate_bands(bands)
        self.bands = bands
        self.band_indices = Tensor(
            [self.all_band_names.index(b) for b in bands if b in self.all_band_names]
        ).long()

        self._verify()

        if split == "train":
            splitfilename = f"eurosat-{split}{train_split_file_suffix}"
        else:
            splitfilename = f"eurosat-{split}.txt"

        valid_fns = set()
        with open(os.path.join(self.root, splitfilename)) as f:
            for fn in f:
                valid_fns.add(fn.strip().replace(".jpg", ".tif"))
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=root,
            transforms=transforms,
            loader=rasterio_loader,
            is_valid_file=is_in_split,
        )

    def __getitem__(self, index: int):
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)

        image = torch.index_select(image, dim=0, index=self.band_indices).float()
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _check_integrity(self) -> bool:
        """Check integrity of dataset.

        Returns:
            True if dataset files are found and/or MD5s match, else False
        """
        integrity: bool = check_integrity(
            os.path.join(self.root, self.filename), self.md5 if self.checksum else None
        )
        return integrity

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the files already exist
        filepath = self.root
        if os.path.exists(filepath):
            return

        # Check if zip file already exists (if so then extract)
        if self._check_integrity():
            self._extract()
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download and extract the dataset
        self._download()
        self._extract()

    def _download(self) -> None:
        """Download the dataset."""
        download_url(
            self.url,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                filename=f"eurosat-{split}.txt",
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _extract(self) -> None:
        """Extract the dataset."""
        filepath = os.path.join(self.root, self.filename)
        extract_archive(filepath)

    def _validate_bands(self, bands: Sequence[str]) -> None:
        """Validate list of bands.

        Args:
            bands: user-provided sequence of bands to load

        Raises:
            AssertionError: if ``bands`` is not a sequence
            ValueError: if an invalid band name is provided

        .. versionadded:: 0.3
        """
        assert isinstance(bands, Sequence), "'bands' must be a sequence"
        for band in bands:
            if band not in self.all_band_names:
                raise ValueError(f"'{band}' is an invalid band name.")

    def plot(
        self,
        sample,
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`NonGeoClassificationDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        Raises:
            ValueError: if RGB bands are not found in dataset

        .. versionadded:: 0.2
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        image = np.take(sample["image"].numpy(), indices=rgb_indices, axis=0)
        image = np.rollaxis(image, 0, 3)
        image = np.clip(image / 3000, 0, 1)

        label = cast(int, sample["label"].item())
        label_class = self.classes[label]

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction = cast(int, sample["prediction"].item())
            prediction_class = self.classes[prediction]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            title = f"Label: {label_class}"
            if showing_predictions:
                title += f"\nPrediction: {prediction_class}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
