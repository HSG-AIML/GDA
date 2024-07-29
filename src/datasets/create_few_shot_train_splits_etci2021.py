import random
import os
import ast
import sys
import shutil
from collections import defaultdict
import torchgeo.datasets


"""
randomly create balanced few-shot training sets for the ETCI2021 torchgeo dataset
call like `python create_few_shot_train_splits.py 10 0`
where 
    argument1 is the number of samples 
    argument2 is the seed for sampling the observations

this will write the new train file in the same directory as the existing split files
"""

if __name__ == "__main__":
    assert len(sys.argv) == 3, sys.argv

    k, seed = sys.argv[1:]

    k = int(k)
    seed = int(seed)
    split = "train"

    random.seed(seed)

    dataset = torchgeo.datasets.ETCI2021(root="data/etci2021")

    samples = dataset.files

    new_samples = random.sample(samples, k)

    split_file_name = f"etci2021-train-k{k}-seed{seed}.txt"

    with open(
        os.path.join(
            dataset.root.replace(dataset.root.split("/")[-1], ""),
            split_file_name,
        ),
        "w",
    ) as f:
        for s in new_samples:
            f.write(str(s) + "\n")
