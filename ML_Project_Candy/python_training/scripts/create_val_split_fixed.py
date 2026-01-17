import os
import shutil
import random
from pathlib import Path

# Paths - relative to python_training root locally
# Adjust if running from different cwd
# Assuming CWD is ML_Project_Candy/python_training
base_dir = Path("datasets/yolo_dataset")
train_img_dir = base_dir / "train/images"
train_lbl_dir = base_dir / "train/labels"
val_img_dir = base_dir / "val/images"
val_lbl_dir = base_dir / "val/labels"

# Ensure val dirs exist
val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# Get list of images - check common extensions
extensions = ['*.jpg', '*.jpeg', '*.png']
images = []
for ext in extensions:
    images.extend(train_img_dir.glob(ext))

print(f"Total training images found: {len(images)}")

if len(images) == 0:
    print("No images found in train directory!")
    exit(1)

# Check if val is already populated
val_images_count = 0
for ext in extensions:
    val_images_count += len(list(val_img_dir.glob(ext)))
    
if val_images_count > 0:
    print(f"Validation directory already has {val_images_count} images. Skipping split.")
    exit(0)

# Calculate split (e.g., 20% for val)
num_val = int(len(images) * 0.2)
if num_val == 0:
    print("Not enough images to split.")
    exit()

val_images = random.sample(images, num_val)

print(f"Moving {len(val_images)} images to validation...")

for img_path in val_images:
    # Move image
    shutil.move(str(img_path), str(val_img_dir / img_path.name))
    
    # Move corresponding label
    label_name = img_path.stem + ".txt"
    label_path = train_lbl_dir / label_name
    
    if label_path.exists():
        shutil.move(str(label_path), str(val_lbl_dir / label_name))
    else:
        print(f"Warning: Label not found for {img_path.name}")

print("Done.")
