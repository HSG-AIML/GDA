import json
import os

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# adapted from https://github.com/isaaccorley/resize-is-all-you-need/


class PadMissingBands:
    def __call__(self, sample):
        B10 = torch.zeros((1, *sample["image"].shape[1:]), dtype=torch.float)
        sample["image"] = torch.cat(
            [sample["image"][:8], B10, sample["image"][8:]], dim=0
        )
        return sample


class TreeSatAI(Dataset):
    classes = [
        "Abies",
        "Acer",
        "Alnus",
        "Betula",
        "Cleared",
        "Fagus",
        "Fraxinus",
        "Larix",
        "Picea",
        "Pinus",
        "Populus",
        "Prunus",
        "Pseudotsuga",
        "Quercus",
        "Tilia",
    ]
    splits = {
        "train": "train_filenames_new.lst",  # note: here, the val set was split out from the original train set (10%)
        "val": "val_filenames.lst",
        "test": "test_filenames.lst",
    }
    sizes = {20: "200m", 6: "60m"}
    labels_path = os.path.join("labels", "TreeSatBA_v9_60m_multi_labels.json")
    s2_all_bands = [
        "B2",
        "B3",
        "B4",
        "B8",
        "B5",
        "B6",
        "B7",
        "B8A",
        "B11",
        "B12",
        "B1",
        "B9",
    ]
    s1_all_bands = ["VV", "VH", "VV/VH"]
    aerial_all_bands = ["NIR", "R", "G", "B"]
    s2_rgb_bands = ["B4", "B3", "B2"]
    s1_rgb_bands = s1_all_bands
    aerial_rgb_bands = ["R", "G", "B"]
    s2_correct_band_order = [
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B9",
        "B11",
        "B12",
    ]

    def __init__(
        self,
        root,
        split="train",
        modality="s1",
        bands=s1_rgb_bands,
        multilabel=False,
        transforms=None,
        size=20,
    ):
        assert split in self.splits
        assert size in self.sizes
        for band in bands:
            assert band in getattr(
                self, f"{modality}_all_bands"
            ), f"Band {band} not in bands of modality {modality}"

        self.modality = modality
        self.band_indices = [
            getattr(self, f"{self.modality}_all_bands").index(band) for band in bands
        ]
        self.band_indices_rasterio = [idx + 1 for idx in self.band_indices]

        self.root = root
        self.split = split
        self.size = size
        self.bands = bands
        self.multilabel = multilabel
        self.transforms = transforms
        self.num_classes = len(self.classes)

        image_root = os.path.join(root, self.modality, self.sizes[size])
        split_path = os.path.join(root, self.splits[split])
        with open(split_path) as f:
            images = f.read().strip().splitlines()
        self.images = [os.path.join(image_root, image) for image in images]

        if self.multilabel:
            labels_path = os.path.join(root, self.labels_path)
            with open(labels_path) as f:
                self.labels = json.load(f)
        else:
            self.labels = [
                os.path.basename(image).split("_")[0] for image in self.images
            ]

    def __len__(self):
        return len(self.images)

    def _load_image(self, path):
        with rasterio.open(path) as f:
            image = f.read(
                self.band_indices_rasterio,  # out_shape=(self.size, self.size)
            )  # .astype(np.int32)

        image = torch.from_numpy(image)
        if self.modality == "s2":
            image = image.to(torch.float).clip(min=0.0, max=None)
        image = image.to(torch.float)

        return image

    def _load_target(self, index):
        if self.multilabel:
            filename = os.path.basename(self.images[index])
            onehot = torch.zeros((self.num_classes,), dtype=torch.float)
            for cls, score in self.labels[filename]:
                idx = self.classes.index(cls)
                onehot[idx] = score
            return onehot
        else:
            cls = self.labels[index]
            label = self.classes.index(cls)
            label = torch.tensor(label).to(torch.long)
            return label

    def __getitem__(self, index):
        path = self.images[index]
        image = self._load_image(path)
        label = self._load_target(index)
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
