import os
import random
import shutil
from pathlib import Path

# Répertoires
base = Path(__file__).parent / "datasets" / "yolo_dataset"
img_dir = base / "images"
label_dir = base / "labels"

# Cibles
splits = ["train", "val", "test"]
percentages = {"train": 0.8, "val": 0.1, "test": 0.1}

# Lister toutes les images
all_images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
random.shuffle(all_images)

total = len(all_images)
train_end = int(percentages["train"] * total)
val_end = train_end + int(percentages["val"] * total)

split_indices = {
    "train": (0, train_end),
    "val": (train_end, val_end),
    "test": (val_end, total)
}

# Créer les dossiers de destination
for split in splits:
    (base / split / "images").mkdir(parents=True, exist_ok=True)
    (base / split / "labels").mkdir(parents=True, exist_ok=True)

# Copier les fichiers dans les bons splits
for split in splits:
    start, end = split_indices[split]
    for img_path in all_images[start:end]:
        label_path = label_dir / (img_path.stem + ".txt")
        dest_img = base / split / "images" / img_path.name
        dest_label = base / split / "labels" / (img_path.stem + ".txt")
        
        shutil.copy2(img_path, dest_img)
        if label_path.exists():
            shutil.copy2(label_path, dest_label)

print(f"Split terminé : {train_end} train, {val_end-train_end} val, {total-val_end} test")
print(f"Total images traitées : {total}")
