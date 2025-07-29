import os
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = r'D:\ml training\new linked'
DEST_DIR = r'D:\ml training\split_dataset'
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train, val, test

def split_data():
    classes = os.listdir(SOURCE_DIR)
    for cls in tqdm(classes, desc="Splitting dataset"):
        class_path = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * SPLIT_RATIOS[0])
        val_end = train_end + int(total * SPLIT_RATIOS[1])

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split, split_images in splits.items():
            split_dir = os.path.join(DEST_DIR, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_images:
                src_img = os.path.join(class_path, img)
                dst_img = os.path.join(split_dir, img)
                shutil.copy2(src_img, dst_img)

if __name__ == "__main__":
    split_data()