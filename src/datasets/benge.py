import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import kornia
import torch
from torch.utils.data import Dataset
import albumentations as A
from kornia.augmentation import GeometricAugmentationBase2D

# from albumentations.pytorch import ToTensorV2
import rasterio as rio
from rasterio.enums import Resampling
import matplotlib as mpl
import matplotlib.pyplot as plt


S2_ALL = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
S2_RGB = ["B04", "B03", "B02"]  # 10m
S2_RED_EDGE = ["B05", "B06", "B07"]  # 20m
S2_SWIR_RE4 = ["B11", "B12", "B8A"]  # 20m swir and red edge band 4
S2_NIR = ["B08"]  # 10m
S2_BAND_GROUPS = {
    "S2_RGB": S2_RGB,
    "S2_RED_EDGE": S2_RED_EDGE,
    "S2_SWIR_RE4": S2_SWIR_RE4,
    "S2_NIR": S2_NIR,
}


class BENGE(Dataset):
    """A dataset class implementing all ben-ge data modalities."""

    def __init__(
        self,
        data_dir=None,
        s2_bands={"ALL": S2_ALL},
        s1_bands={"S1_VH": "VH", "S1_VV": "VV"},
        s2_native_res=False,
        lulc=True,
        dem=None,
        split="train",
        few_shot_k=None,
        few_shot_seed=None,
        transforms=None,
    ):
        """Dataset class constructor

        keyword arguments:
        data_dir -- string containing the path to the base directory of ben-ge dataset, default: ben-ge-800 directory
        s2_bands -- list of Sentinel-2 bands to be extracted, default: all bands
        s1_bands -- list of Senintel-1 bands to be extracted, default: all bands

        returns:
        BENGE object
        """
        super().__init__()

        # store some definitions
        self.data_dir = data_dir
        self.s2_bands = s2_bands
        self.s1_bands = s1_bands
        self.s2_native_res = s2_native_res
        self.lulc = lulc
        self.dem = dem
        self.transforms = transforms
        # self.augmentation = A.Compose(
        #     [
        #         ToTensorV2(),
        #     ]
        # )

        # read in relevant data files and definitions
        self.name = self.data_dir.split("/")[-1]
        self.meta = pd.read_csv(f"{self.data_dir}/{self.name}_meta.csv")
        if few_shot_k is not None and few_shot_seed is not None and split == "train":
            split_file = f"{self.data_dir}/splits/{self.name}_{split}_k{few_shot_k}_seed{few_shot_seed}.csv"
        else:
            split_file = f"{self.data_dir}/splits/{self.name}_{split}.csv"
        self.split_patch_ids = pd.read_csv(
            split_file,
            header=None,
            names=["patch_id"],
        )
        # read only data from specified split
        self.meta = self.meta[self.meta.patch_id.isin(self.split_patch_ids.patch_id)]
        self.ewc_labels = pd.read_csv(f"{self.data_dir}/{self.name}_esaworldcover.csv")
        self.ewc_labels = self.ewc_labels[
            self.ewc_labels.patch_id.isin(self.split_patch_ids.patch_id)
        ]
        self.ewc_label_names = [
            "tree_cover",
            "shrubland",
            "grassland",
            "cropland",
            "built-up",
            "bare/sparse_vegetation",
            "snow_and_ice",
            "permanent_water_bodies",
            "herbaceous_wetland",
            "mangroves",
            "moss_and_lichen",
        ]
        self.classes = self.ewc_label_names
        self.s2_resampling_factors = {
            "B01": 6,
            "B02": 1,
            "B03": 1,
            "B04": 1,
            "B05": 2,
            "B06": 2,
            "B07": 2,
            "B08": 1,
            "B09": 6,
            "B11": 2,
            "B12": 2,
            "B8A": 2,
        }

    def __getitem__(self, idx, lulc_mask=False):
        """Return sample `idx` as dictionary from the dataset."""
        sample_info = self.meta.iloc[idx]
        patch_id = sample_info.patch_id  # extract Sentinel-2 patch id
        patch_id_s1 = sample_info.patch_id_s1  # extract Sentinel-1 patch id
        lulc_mask = lulc_mask or self.lulc

        sample = {}

        if self.s2_bands:
            s2 = self.load_s2(patch_id)  # load Sentinel-2 data
            assert len(s2) == 1
            for name, bands in s2.items():
                sample["image"] = bands
        if self.s1_bands:
            s1 = self.load_s1(patch_id_s1)  # load Sentinel-1 data
            for k in list(s1.keys()):
                s1[k] = (
                    np.clip(s1[k].astype(float), a_min=-25, a_max=0.0) + 25.0
                ) / 25.0
            sample["image"] = np.stack(
                list(s1.values()), axis=-1
            ).squeeze()  # stack vv and vh bands
            sample["image"] = np.concatenate(
                [sample["image"], np.expand_dims(sample["image"][:, :, 0], -1)], axis=-1
            )  # make it 3 channels
        else:
            s1 = None

        # extract top land-use/land-cover label
        label = np.argmax(
            self.ewc_labels[self.ewc_labels.patch_id == patch_id][self.ewc_label_names]
        )
        sample["label"] = label

        if lulc_mask:
            # land-use/land-cover map data
            ewc_mask = self.load_ewc(patch_id).astype(float)
            ewc_mask[ewc_mask == 100] = 110
            ewc_mask[ewc_mask == 95] = 100
            ewc_mask = ewc_mask / 10 - 1  # transform to scale [0, 11]
            ewc_mask = np.moveaxis(ewc_mask, 0, -1)
            sample["mask"] = ewc_mask.squeeze()
        else:
            ewc_mask = None

        # for img in s2.values():
        # augmented = self.augmentation(
        # image=img,  # mask=ewc_mask
        # )  # generate augmented data

        # reassign and normalize augmented Sentinel-2 data
        # img = torch.clip(augmented["image"].float() / 10000, 0, 1)

        # if lulc_mask:
        # reassign augmented land-use/land-cover data
        #     ewc_mask = augmented["mask"]

        if self.dem is not None:
            dem = self.load_dem(patch_id)
            dem = np.moveaxis(dem, 0, -1)
            sample["image"] = dem
        else:
            dem = None

        season = sample_info["season_s2"]  # seasonal data
        climatezone = {
            0: 0,
            7: 1,
            8: 2,
            9: 3,
            14: 4,
            15: 5,
            16: 6,
            18: 7,
            25: 8,
            26: 9,
            27: 10,
            29: 11,
        }[
            sample_info["climatezone"]
        ]  # climatezone data

        # create sample dictionary containing all the data
        # sample = {
        #     "patch_id": patch_id,
        #     # "lulc_top": torch.from_numpy(np.array([label.copy()], dtype=float)),
        #     # "lulc_top": np.array([label.copy()], dtype=float),
        #     "season": season,
        #     "climatezone": climatezone,
        # }
        # sample.update(s2)  # add the sentinel2 band groups

        # if s1 is not None:  # add Sentinel-1 data, if generated
        #     sample.update(s1)  # torch.tensor(s1).float()
        # if ewc_mask is not None:
        #     sample["lulc"] = ewc_mask
        # if dem is not None:
        #     sample["dem"] = dem
        sample["image"] = np.moveaxis(sample["image"], -1, 0)  # C,H,W
        for k, v in sample.items():
            sample[k] = torch.tensor(v, dtype=torch.float)
        sample["label"] = sample["label"].to(torch.long)
        sample["mask"] = sample["mask"].to(torch.long).unsqueeze(0).unsqueeze(0)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        """Return length of this dataset."""
        return self.meta.shape[0]

    def load_s2(self, patch_id):
        """Helper function to load Sentinel-2 data for a given `patch_id`."""

        imgs = {}
        for group_name, bands in self.s2_bands.items():
            img = []
            for band in bands:
                # read corresponding data file and upsample based on resampling factor
                with rio.open(
                    f"{self.data_dir}/sentinel-2/{patch_id}/{patch_id}_{band}.tif"
                ) as d:
                    if self.s2_native_res:
                        img.append(d.read())
                    else:
                        # resample all bands to 10m resolution
                        upscale_factor = self.s2_resampling_factors.get(band)
                        data = d.read(
                            out_shape=(
                                d.count,
                                int(d.height * upscale_factor),
                                int(d.width * upscale_factor),
                            ),
                            resampling=Resampling.bilinear,
                        )
                        img.append(data)

            img = np.concatenate(img).astype(float)
            img = np.moveaxis(img, 0, -1)

            imgs[group_name] = img

        return imgs

    def load_dem(self, dem_patch_id):
        with rio.open(f"{self.data_dir}/dem/{dem_patch_id}_dem.tif") as d:
            data = d.read().astype(float)

        return data

    def load_s1(self, s1_patch_id):
        """Helper function to load Sentinel-1 data for a given `patch_id`."""
        img = {}

        for name, band in self.s1_bands.items():
            # read corresponding data file
            with rio.open(
                f"{self.data_dir}/sentinel-1/{s1_patch_id}/{s1_patch_id}_{band}.tif"
            ) as d:
                data = d.read()
                img[name] = np.moveaxis(data, 0, -1).astype(float)

        return img

    def load_ewc(self, patch_id):
        """Helper function to load ESAWorldCover data for a given `patch_id`."""
        with rio.open(
            f"{self.data_dir}/esaworldcover/{patch_id}_esaworldcover.tif"
        ) as d:
            data = d.read()

        return data

    def visualize_observation(self, idx):
        """Visualize data sample `idx`."""

        # define ESA WorldCover colormap
        COLOR_CATEGORIES = [
            (0, 100, 0),
            (255, 187, 34),
            (255, 255, 76),
            (240, 150, 255),
            (250, 0, 0),
            (180, 180, 180),
            (240, 240, 240),
            (0, 100, 200),
            (0, 150, 160),
            (0, 207, 117),
            (250, 230, 160),
        ]
        cmap_all = mpl.colors.ListedColormap(np.array(COLOR_CATEGORIES) / 255.0)

        # read sample
        sample = self.__getitem__(idx)
        image = sample.get("s2").squeeze()
        mask = sample.get("lulc_mask")

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        img_rgb = image[[3, 2, 1], :, :]
        img_rgb = np.transpose(img_rgb, (1, 2, 0))
        scaled_img_rgb = (img_rgb - np.min(img_rgb.numpy(), axis=(0, 1))) / (
            np.max(img_rgb.numpy(), axis=(0, 1)) - np.min(img_rgb.numpy(), axis=(0, 1))
        )

        axs[0].imshow(np.clip(1.5 * scaled_img_rgb + 0.1, 0, 1))
        axs[0].set_title("Sentinel-2 RGB")
        axs[0].axis("off")

        mask = mask.squeeze()

        axs[1].imshow(mask, cmap=cmap_all, vmin=0, vmax=11, interpolation=None)
        axs[1].set_title(
            " Climate Zone: {} \n Season: {} \n Segmentation Mask".format(
                sample.get("climatezone"), sample.get("season")
            )
        )
        axs[1].axis("off")

        plt.tight_layout()
        plt.show()
        return

    def visualise_predictions(self, idx, predictions):
        """Visualize data sample `idx` and corresponding `predictions`."""
        COLOR_CATEGORIES = [
            (0, 100, 0),
            (255, 187, 34),
            (255, 255, 76),
            (240, 150, 255),
            (250, 0, 0),
            (180, 180, 180),
            (240, 240, 240),
            (0, 100, 200),
            (0, 150, 160),
            (0, 207, 117),
            (250, 230, 160),
        ]

        season, climate_zone, land_cover = predictions

        cmap_all = mpl.colors.ListedColormap(np.array(COLOR_CATEGORIES) / 255.0)

        sample = self.__getitem__(idx)
        image = sample.get("s2").squeeze()
        mask = sample.get("lulc_mask")

        fig, axs = plt.subplots(1, 3, figsize=(12, 6))

        img_rgb = image[[3, 2, 1], :, :]
        img_rgb = np.transpose(img_rgb, (1, 2, 0))
        scaled_img_rgb = (img_rgb - np.min(img_rgb.numpy(), axis=(0, 1))) / (
            np.max(img_rgb.numpy(), axis=(0, 1)) - np.min(img_rgb.numpy(), axis=(0, 1))
        )

        axs[0].imshow(np.clip(1.5 * scaled_img_rgb + 0.1, 0, 1))
        axs[0].set_title("Sentinel-2 RGB")
        axs[0].axis("off")

        mask = mask.squeeze()

        axs[2].imshow(mask, cmap=cmap_all, vmin=0, vmax=11)
        axs[2].set_title(
            " Groundtruths \n Climate Zone: {} \n Season: {} \n Segmentation Mask".format(
                sample.get("climatezone"), sample.get("season")
            )
        )
        axs[2].axis("off")

        axs[1].imshow(land_cover, cmap=cmap_all, vmin=0, vmax=11)
        axs[1].set_title(
            " Predictions \n Climate Zone: {} \n Season: {} \n Segmentation Mask".format(
                climate_zone, season
            )
        )
        axs[1].axis("off")

        plt.show()
        return


class BENGESampleCollateFn:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, samples):
        imgs = []
        for img in samples.values():
            if isinstance(img, np.ndarray):
                # transform raster data
                img = kornia.utils.image_to_tensor(img)
                img = self.transforms(img)
                img = img.contiguous().float()
                imgs.append(img)

        return imgs


class KorniaPrepAugmentation(GeometricAugmentationBase2D):
    def __init__(
        self,
        same_on_batch: bool = False,
        p: float = 1.0,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

    def apply_transform(self, input, params):
        input = torch.tensor(input)
        input = torch.moveaxis(input, -1, 0)
        if "S2" in params["modality"]:
            input /= 10000

        if "S1" in params["modality"]:
            input = torch.nan_to_num(input)
            input = torch.clip(input, -25, 0)
            input /= 25
            input += 1

        return input
