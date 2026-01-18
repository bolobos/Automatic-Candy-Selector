#!/usr/bin/env python3
"""
Script pour créer le dossier de validation à partir des images de Test
"""
import os
import shutil
from pathlib import Path
import random

# Chemins
test_dir = Path('../datasets/nos_dataset/Test')
train_img_dir = Path('../datasets/yolo_dataset/train/images')
train_lbl_dir = Path('../datasets/yolo_dataset/train/labels')
val_img_dir = Path('datasets/yolo_dataset/val/images')
val_lbl_dir = Path('datasets/yolo_dataset/val/labels')
nos_test_dir = Path('datasets/nos_dataset/Test')

# Créer dossiers validation
val_img_dir.mkdir(parents=True, exist_ok=True)
val_lbl_dir.mkdir(parents=True, exist_ok=True)

# Lister toutes les images Test (sauf lebron)
test_images = [f for f in test_dir.glob('*.jpg') if 'lebron' not in f.name.lower()]
test_images.extend([f for f in test_dir.glob('*.JPG') if 'lebron' not in f.name.lower()])

print(f'Images trouvées dans Test (sans lebron): {len(test_images)}')

# Copier images de Test vers val
copied = 0
for img_path in test_images:
    # Chercher le label correspondant dans train
    img_stem = img_path.stem
    label_file = train_lbl_dir / f'{img_stem}.txt'
    
    # Si l'image existe dans train, la copier vers val
    train_img = train_img_dir / img_path.name
    if train_img.exists():
        # Copier image
        dest_img = val_img_dir / img_path.name
        if not dest_img.exists():
            shutil.copy2(train_img, dest_img)
        # Copier label si existe
        if label_file.exists():
            dest_lbl = val_lbl_dir / f'{img_stem}.txt'
            if not dest_lbl.exists():
                shutil.copy2(label_file, dest_lbl)
        copied += 1
        print(f"  Copié: {img_path.name}")

print(f'\nImages copiées de train vers val: {copied}')

# Ajouter images aléatoires si besoin (minimum 15% du train)
train_images = list(train_img_dir.glob('*.jpg'))
target_val_size = max(30, int(len(train_images) * 0.15))
current_val_size = len(list(val_img_dir.glob('*.jpg')))

print(f"Taille actuelle validation: {current_val_size}")
print(f"Taille cible validation: {target_val_size}")

if current_val_size < target_val_size:
    needed = target_val_size - current_val_size
    print(f'Ajout de {needed} images supplémentaires...')
    
    # Images disponibles (pas déjà dans val)
    val_stems = {f.stem for f in val_img_dir.glob('*.jpg')}
    available = [img for img in train_images if img.stem not in val_stems]
    
    # Sélection aléatoire
    random.seed(42)
    to_add = random.sample(available, min(needed, len(available)))
    
    for img_path in to_add:
        # Copier vers val
        shutil.copy2(img_path, val_img_dir / img_path.name)
        label_file = train_lbl_dir / f'{img_path.stem}.txt'
        if label_file.exists():
            shutil.copy2(label_file, val_lbl_dir / f'{img_path.stem}.txt')
        
        # Copier aussi vers nos_dataset/Test
        dest_test = nos_test_dir / img_path.name
        if not dest_test.exists():
            shutil.copy2(img_path, dest_test)
            print(f"  Ajouté à validation ET Test: {img_path.name}")

final_count = len(list(val_img_dir.glob('*.jpg')))
final_count_lbl = len(list(val_lbl_dir.glob('*.txt')))
print(f'\n✅ Total images validation: {final_count}')
print(f'✅ Total labels validation: {final_count_lbl}')
