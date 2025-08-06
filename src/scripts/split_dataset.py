import os
import shutil
import random
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
SOURCE_DIR = "garbage-dataset-5"  # where all original class folders are
TARGET_DIR = "dataset"          # where split folders go
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, val, test

# -----------------------------
# RUN
# -----------------------------
def split_dataset(source_dir, target_dir, split_ratios):
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(split_ratios[0] * n_total)
        n_val = int(split_ratios[1] * n_total)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, image_list in splits.items():
            split_dir = os.path.join(target_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in image_list:
                src = os.path.join(cls_path, img_name)
                dst = os.path.join(split_dir, img_name)
                shutil.copy2(src, dst)

    print(f"Dataset split complete. Output in: {target_dir}")

if __name__ == "__main__":
    split_dataset(SOURCE_DIR, TARGET_DIR, SPLIT_RATIOS)
