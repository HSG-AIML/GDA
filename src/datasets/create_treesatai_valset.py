import os
import shutil
import random
import glob

# split a validation set from the original 'train' set.

if __name__ == "__main__":
    random.seed(0)

    names = [
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

    with open(
        "data/treesatai/test_filenames.lst",
        "r",
    ) as f:
        files = [l.strip() for l in f.readlines()]

    with open(
        "data/treesatai/train_filenames.lst",
        "r",
    ) as f:
        all_train_files = [l.strip() for l in f.readlines()]

    print(f"{len(files)=}, {len(all_train_files)=}")
    with open(
        "data/treesatai/train_filenames_new.lst",
        "w",
    ) as new_train_file:
        with open(
            "data/treesatai/val_filenames.lst",
            "w",
        ) as val_file:
            for c in names:
                class_files = [f for f in files if f.startswith(c)]
                train_files = [f for f in all_train_files if f.startswith(c)]

                all_idx = set(range(len(train_files)))
                chosen_idx = set(random.sample(all_idx, len(class_files)))
                remaining_idx = all_idx - chosen_idx

                for idx in chosen_idx:
                    val_file.write(train_files[idx] + "\n")

                for idx in remaining_idx:
                    new_train_file.write(train_files[idx] + "\n")
