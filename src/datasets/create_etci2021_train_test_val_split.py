import random
import numpy as np
from torchgeo.datamodules import ETCI2021DataModule

"""
ETCI2021 comes with pre-defined train/val split, but the provided test split only has images and no labels (masks)
This script splits the original torchgeo val set into half, one for validation, one for testing. After that, the original
test set (without labels) is replaced with the new test set that has labels."""

if __name__ == "__main__":
    seed = 0
    random.seed(seed)

    dm = ETCI2021DataModule(root="data/etci2021")
    dm.setup("fit")

    files = np.array(dm.val_dataset.files)

    indices = list(range(0, len(files)))
    random.shuffle(indices)
    indices = np.array(indices)

    val_indices = indices[: len(indices) // 2]
    test_indices = indices[len(indices) // 2 :]

    valset = files[val_indices].tolist()
    testset = files[test_indices].tolist()

    with open("data/etci2021-val.txt", "w") as f:
        for line in valset:
            f.write(str(line) + "\n")

    with open("data/etci2021-test.txt", "w") as f:
        for line in testset:
            f.write(str(line) + "\n")
