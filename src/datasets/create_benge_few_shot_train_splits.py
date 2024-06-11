import os
import pandas as pd
import numpy as np


if __name__ == "__main__":
    k = 40
    seed = 2

    split = "train"
    name = "ben-ge-8k"
    data_dir = "/netscratch/lscheibenreif/ben-ge-8k/"
    meta = pd.read_csv(f"{data_dir}/{name}_meta.csv")
    split_patch_ids = pd.read_csv(
        f"{data_dir}/splits/{name}_{split}.csv",
        header=None,
        names=["patch_id"],
    )

    # read only data from specified split
    meta = meta[meta.patch_id.isin(split_patch_ids.patch_id)]
    ewc_labels = pd.read_csv(f"{data_dir}/{name}_esaworldcover.csv")
    ewc_labels = ewc_labels[ewc_labels.patch_id.isin(split_patch_ids.patch_id)]
    ewc_label_names = [
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

    ewc_labels["label"] = [
        np.argmax(ewc_labels[ewc_labels.patch_id == patch_id][ewc_label_names])
        for patch_id in ewc_labels.patch_id.tolist()
    ]

    few_shot_patch_ids = []
    for label in ewc_labels.label.unique():
        few_shot_patch_ids.extend(
            ewc_labels[ewc_labels.label == label]
            .sample(k, replace=True, random_state=seed)
            .patch_id.tolist()
        )

    with open(
        os.path.join(data_dir, "splits", f"{name}_train_k{k}_seed{seed}.csv"), "w"
    ) as f:
        for line in few_shot_patch_ids:
            f.write(line + "\n")
