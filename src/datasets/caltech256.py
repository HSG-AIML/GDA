import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets
from PIL import Image

# transform_train = transforms.Compose([
#            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class Caltech256Dataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = root
        self.data = torchvision.datasets.Caltech256(root=root)
        self.categories = np.array(self.data.categories[:-1])  # without clutter class
        self.full_category = {x.split(".")[0]: x for x in self.categories}
        self.classes = self.categories
        self.category_string_map = {
            idx: self.categories[idx].split(".")[0]
            for idx in range(len(self.categories))
        }
        self.string_category_map = {v: k for k, v in self.category_string_map.items()}
        self.split = split
        self.transforms = transforms
        with open(root + f"/caltech256-{split}.txt", "r") as f:
            self.files = [x.strip() for x in f.readlines()]
        self.labels = [self.string_category_map[x.split("_")[0]] for x in self.files]

    def __getitem__(self, idx):
        file = self.files[idx]

        image = Image.open(
            os.path.join(
                self.root,
                "caltech256",
                "256_ObjectCategories",
                self.full_category[file.split("_")[0]],
                file,
            )
        )
        label = self.string_category_map[file.split("_")[0]]
        image = (
            torchvision.transforms.functional.pil_to_tensor(image).to(torch.float)
            / 255.0
        )

        if image.shape[0] == 1:
            # some images are grayscale
            image = torch.stack([image, image, image]).squeeze()

        sample = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.files)
