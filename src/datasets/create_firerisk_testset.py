import os
import shutil
import random
import glob

# split the 'val' set (21k images) randomly in half to create a test set
# roughly preserves the class imbalance of the val set in the test set


if __name__ == "__main__":
    random.seed(0)

    names = [
        "High",
        "Low",
        "Moderate",
        "Non-burnable",
        "Very_High",
        "Very_Low",
        "Water",
    ]

    for c in names:
        filenames = glob.glob(
            os.path.join(
                f"data/FireRisk/val/{c}/*.png"
            )
        )
        chosen_files = random.sample(filenames, int(0.5 * len(filenames)))

        for file in chosen_files:
            shutil.move(file, file.replace("val", "test"))
