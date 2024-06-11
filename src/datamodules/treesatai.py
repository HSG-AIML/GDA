import kornia.augmentation as K
import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.datasets import TreeSatAI
from src.datasets.treesatai import PadMissingBands

# adapted from https://github.com/isaaccorley/resize-is-all-you-need/


class TreeSatAIDataModule(LightningDataModule):
    # stats computes on the new trainset, after splitting out val set
    s1_mean = torch.tensor([-3.4632447, -9.345917, 0.39350712])  # vv,vh,vv/vh
    s1_std = torch.tensor([2.341043, 2.5721953, 0.34939596])
    aerial_mean = torch.tensor(
        [114.54107514, 99.08938928, 79.31825658, 85.8420187]
    )  # nir,r,g,b
    aerial_std = torch.tensor([42.19899775, 37.99084927, 32.33622661, 40.03827531])
    # stats computed uses TreeSatAI.correct_band_order
    s2_min = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # padded band
    )
    s2_max = torch.tensor(
        [
            5341.0,
            18533.0,
            17403.0,
            16670.0,
            16369.0,
            16027.0,
            15855.0,
            15727.0,
            15679.0,
            10354.0,
            0.0,  # padded band
            15202.0,
            15098.0,
        ]
    )
    s2_mean = (
        torch.tensor(
            [
                253.16619873046875,
                237.94383239746094,
                384.6624450683594,
                253.626708984375,
                625.8056640625,
                2074.90234375,
                2651.58642578125,
                2762.07763671875,
                2923.10107421875,
                2926.369140625,
                0.0,  # padded band
                1307.518310546875,
                600.5294189453125,
            ]
        )
        / 10000.0
    )
    s2_std = (
        torch.tensor(
            [
                127.59476470947266,
                137.15211486816406,
                160.50038146972656,
                178.5304718017578,
                233.0775604248047,
                542.3641357421875,
                721.9783935546875,
                798.6138305664062,
                790.643798828125,
                717.2382202148438,
                10000.0,  # padded band
                462.2733154296875,
                300.0897216796875,
            ]
        )
        / 10000.0
    )

    s2_norm_rgb = K.Normalize(mean=s2_mean[[3, 2, 1]], std=s2_std[[3, 2, 1]])
    s2_norm_msi = K.Normalize(mean=s2_mean, std=s2_std)

    @staticmethod
    def preprocess(sample):
        sample["image"] = sample["image"].float()
        return sample

    def __init__(
        self,
        root,
        modality="s1",
        bands=["VV", "VH", "VV/VH"],
        multilabel=False,
        size=6,
        pad_missing_bands=False,
        batch_size=32,
        num_workers=8,
        seed=0,
        transforms=None,
    ):
        self.root = root
        self.modality = modality
        self.bands = bands
        self.multilabel = multilabel
        self.batch_size = batch_size
        self.size = size
        self.num_workers = num_workers
        self.pad_missing_bands = pad_missing_bands
        self.generator = torch.Generator().manual_seed(seed)
        self.transforms = transforms

        if self.modality == "aerial":
            assert (
                self.size == 6
            ), f"Aerial imagery not available for size {self.size}, only for size=6"

    def setup(self, fit):
        # transforms = [TreeSatAIDataModule.preprocess]
        # if self.pad_missing_bands:
        # transforms.append(PadMissingBands())

        self.train_dataset = TreeSatAI(
            root=self.root,
            split="train",
            modality=self.modality,
            bands=self.bands,
            multilabel=self.multilabel,
            # transforms=T.Compose(transforms),
            transforms=self.transforms,
            size=self.size,
        )
        self.val_dataset = TreeSatAI(
            root=self.root,
            split="val",
            modality=self.modality,
            bands=self.bands,
            multilabel=self.multilabel,
            # transforms=T.Compose(transforms),
            transforms=self.transforms,
            size=self.size,
        )
        self.test_dataset = TreeSatAI(
            root=self.root,
            split="test",
            modality=self.modality,
            bands=self.bands,
            multilabel=self.multilabel,
            # transforms=T.Compose(transforms),
            transforms=self.transforms,
            size=self.size,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
