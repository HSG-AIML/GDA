import random
import os
import sys
import shutil
from collections import defaultdict
import torchgeo.datasets


"""
randomly create balanced few-shot training sets for torchgeo datasets
call like `python create_few_shot_train_splits.py UCMerced 10 0`
where 
    argument1 is the name of the dataset in torchgeo (UCMerced, EuroSAT, ...)
    argument2 is the number of samples per class (1, 5, 10, ...)
    argument3 is the seed for sampling the observations

this will write the new train file in the same directory as the existing split files
"""

if __name__ == "__main__":
    assert len(sys.argv) == 4, sys.argv

    dataset_name, k, seed = sys.argv[1:]

    k = int(k)
    seed = int(seed)
    split = "train"

    random.seed(seed)

    dataset = torchgeo.datasets.__dict__[dataset_name](root="data/")
    class_path_map = defaultdict(list)

    for path, c in dataset.samples:
        class_path_map[c].append(path)

    new_samples = []
    for class_name, files in class_path_map.items():
        new_samples.extend(random.sample(files, k))

    split_file_name = (
        dataset.split_urls[split].split("/")[-1].replace("-train.txt", "")
    )  # e.g., 'uc_merced' for UCMerced dataset

    if hasattr(dataset, "base_dir"):
        # eurosat, ucmerced
        base = dataset.base_dir
    else:
        # resisc45
        base = dataset.directory

    assert len(new_samples) == k * len(class_path_map), "not k samples per class"

    with open(
        os.path.join(
            dataset.root.replace(base, ""),
            f"{split_file_name}-{split}-k{k}-seed{seed}.txt",
        ),
        "w",
    ) as f:
        for s in new_samples:
            f.write(f"{s.split('/')[-1]}\n")
