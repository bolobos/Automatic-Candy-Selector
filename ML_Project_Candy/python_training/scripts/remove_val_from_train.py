#!/usr/bin/env python3
"""
Supprime les images de validation du dossier train pour éviter le data leakage
"""
import os
from pathlib import Path

val_img_dir = Path('datasets/yolo_dataset/val/images')
val_lbl_dir = Path('datasets/yolo_dataset/val/labels')
train_img_dir = Path('datasets/yolo_dataset/train/images')
train_lbl_dir = Path('datasets/yolo_dataset/train/labels')

# Liste des images dans val
val_images = {f.name for f in val_img_dir.glob('*.jpg')}
val_labels = {f.name for f in val_lbl_dir.glob('*.txt')}

print(f'Images à supprimer de train: {len(val_images)}')

# Supprimer les images de train qui sont dans val
deleted_img = 0
deleted_lbl = 0

for img_name in val_images:
    train_img = train_img_dir / img_name
    if train_img.exists():
        train_img.unlink()
        deleted_img += 1
    
    # Supprimer aussi le label correspondant
    lbl_name = Path(img_name).stem + '.txt'
    train_lbl = train_lbl_dir / lbl_name
    if train_lbl.exists():
        train_lbl.unlink()
        deleted_lbl += 1

print(f'✅ Supprimé {deleted_img} images de train')
print(f'✅ Supprimé {deleted_lbl} labels de train')

# Vérification finale
final_train = len(list(train_img_dir.glob('*.jpg')))
final_val = len(list(val_img_dir.glob('*.jpg')))
total = final_train + final_val

print(f'\n📊 Résultat final:')
print(f'   Train: {final_train} images ({final_train/total*100:.1f}%)')
print(f'   Val: {final_val} images ({final_val/total*100:.1f}%)')
print(f'   Total: {total} images')
