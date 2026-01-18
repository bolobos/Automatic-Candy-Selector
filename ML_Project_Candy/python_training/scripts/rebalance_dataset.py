#!/usr/bin/env python3
"""
Script pour analyser et rééquilibrer la distribution des classes dans train/val/test
"""

from pathlib import Path
import shutil
from collections import defaultdict
import random

# Répertoire du script
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

DATASET_DIR = PROJECT_DIR / "datasets" / "yolo_dataset"
CLASSES = ["Tagada", "Dragibus", "Ourson", "Oeuf", "Croco", "Schtroumpf"]

def count_annotations(labels_dir):
    """Compte le nombre d'annotations par classe dans un dossier de labels"""
    class_counts = defaultdict(int)
    image_annotations = {}  # {image_name: [class_ids]}
    
    for label_file in labels_dir.glob("*.txt"):
        annotations = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    annotations.append(class_id)
        
        if annotations:
            image_annotations[label_file.stem] = annotations
    
    return class_counts, image_annotations

def analyze_distribution():
    """Analyse la distribution des classes dans chaque split"""
    print("\n" + "="*60)
    print("📊 ANALYSE DE LA DISTRIBUTION DES CLASSES")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        labels_dir = DATASET_DIR / split / "labels"
        if not labels_dir.exists():
            print(f"\n⚠️  {split} n'existe pas")
            continue
        
        class_counts, _ = count_annotations(labels_dir)
        total = sum(class_counts.values())
        
        print(f"\n{split.upper()}:")
        print(f"  Total annotations: {total}")
        for class_id, count in sorted(class_counts.items()):
            class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"Class {class_id}"
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    {class_name:12s}: {count:3d} ({percentage:5.1f}%)")
    
    print("\n" + "="*60)

def rebalance_dataset():
    """Rééquilibre le dataset pour avoir une distribution plus uniforme dans val"""
    print("\n🔄 RÉÉQUILIBRAGE DU DATASET...\n")
    
    # Analyser la distribution actuelle
    train_counts, train_images = count_annotations(DATASET_DIR / "train" / "labels")
    val_counts, val_images = count_annotations(DATASET_DIR / "val" / "labels")
    test_counts, test_images = count_annotations(DATASET_DIR / "test" / "labels")
    
    # Target: ~20-25 annotations par classe dans val (moyenne)
    target_val = 22
    
    # Créer des listes d'images par classe dominante
    train_by_class = defaultdict(list)
    val_by_class = defaultdict(list)
    
    for img_name, annotations in train_images.items():
        # Classe dominante = celle qui apparaît le plus
        dominant = max(set(annotations), key=annotations.count)
        train_by_class[dominant].append(img_name)
    
    for img_name, annotations in val_images.items():
        dominant = max(set(annotations), key=annotations.count)
        val_by_class[dominant].append(img_name)
    
    # Stratégie de rééquilibrage
    moves = []  # [(image, from_split, to_split, class)]
    
    for class_id in range(len(CLASSES)):
        current_val = val_counts.get(class_id, 0)
        
        if current_val > target_val * 1.5:  # Trop dans val
            # Déplacer vers train
            excess = int(current_val - target_val)
            images_to_move = val_by_class.get(class_id, [])[:excess]
            for img in images_to_move:
                moves.append((img, 'val', 'train', class_id))
        
        elif current_val < target_val * 0.6:  # Pas assez dans val
            # Ramener de train
            needed = int(target_val - current_val)
            images_to_move = train_by_class.get(class_id, [])[:needed]
            for img in images_to_move:
                moves.append((img, 'train', 'val', class_id))
    
    if not moves:
        print("✅ Le dataset est déjà équilibré!")
        return
    
    print(f"📦 {len(moves)} images à déplacer:\n")
    for img, from_split, to_split, class_id in moves:
        print(f"  {CLASSES[class_id]:12s}: {img} ({from_split} → {to_split})")
    
    # Confirmer
    print(f"\n⚠️  Voulez-vous effectuer ces déplacements? (y/n)")
    # Pour automatiser, on accepte automatiquement
    response = 'y'
    
    if response.lower() == 'y':
        for img, from_split, to_split, class_id in moves:
            # Déplacer l'image
            src_img = DATASET_DIR / from_split / "images" / f"{img}.jpg"
            dst_img = DATASET_DIR / to_split / "images" / f"{img}.jpg"
            
            if src_img.exists():
                shutil.move(str(src_img), str(dst_img))
            
            # Déplacer le label
            src_label = DATASET_DIR / from_split / "labels" / f"{img}.txt"
            dst_label = DATASET_DIR / to_split / "labels" / f"{img}.txt"
            
            if src_label.exists():
                shutil.move(str(src_label), str(dst_label))
        
        print("\n✅ Rééquilibrage terminé!")
    else:
        print("\n❌ Rééquilibrage annulé")

if __name__ == "__main__":
    # Analyser d'abord
    analyze_distribution()
    
    # Demander si on veut rééquilibrer
    print("\n🔄 Voulez-vous rééquilibrer le dataset? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        rebalance_dataset()
        print("\n📊 Nouvelle distribution:")
        analyze_distribution()
