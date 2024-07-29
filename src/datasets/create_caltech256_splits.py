import os
import random
from collections import defaultdict
import numpy as np
import torchvision.datasets

train_fraction = 0.9
val_fraction = 0.05
test_fraction = 0.05

assert train_fraction + val_fraction + test_fraction == 1

random.seed(0)

ds = torchvision.datasets.Caltech256(root="data", download=False)

samples = defaultdict(list)
count = 0
for category in os.listdir("data/caltech256/256_ObjectCategories"):
    if category == "257.clutter":
        # remove clutter class
        continue
    for sample in os.listdir(
        os.path.join(
            "data/caltech256/256_ObjectCategories/",
            category,
        )
    ):
        samples[category].append(sample)
        count += 1

print(f"Total number of used samples: {count:,}")


with open("data/caltech256-all.txt", "w") as all_writer:
    with open("data/caltech256-train.txt", "w") as train_writer:
        with open(
            "data/caltech256-val.txt",
            "w",
        ) as val_writer:
            with open(
                "data/caltech256-test.txt",
                "w",
            ) as test_writer:
                for category, cat_samples in samples.items():
                    for sample in cat_samples:
                        all_writer.write(sample + "\n")

                    random.shuffle(cat_samples)

                    for sample in cat_samples[: int(len(cat_samples) * train_fraction)]:
                        train_writer.write(sample + "\n")
                    for sample in cat_samples[
                        int(len(cat_samples) * train_fraction) : int(
                            len(cat_samples) * train_fraction
                        )
                        + int(len(cat_samples) * val_fraction)
                    ]:
                        val_writer.write(sample + "\n")
                    for sample in cat_samples[
                        int(len(cat_samples) * train_fraction)
                        + int(len(cat_samples) * val_fraction) :
                    ]:
                        test_writer.write(sample + "\n")
