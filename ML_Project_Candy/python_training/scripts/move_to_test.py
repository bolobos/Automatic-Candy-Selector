#!/usr/bin/env python3
"""
Script pour déplacer aléatoirement 3 images de chaque catégorie
du dataset d'entraînement vers le dataset de test
"""

import os
import random
import shutil
from pathlib import Path

# Chemins (depuis python_training/)
BASE_DIR = Path(__file__).parent.parent / "datasets"
TRAIN_DIR = BASE_DIR / "nos_dataset" / "Entrainement"
TEST_DIR = BASE_DIR / "nos_dataset" / "Test"
YOLO_TRAIN_IMG = BASE_DIR / "yolo_dataset" / "images" / "train"
YOLO_TRAIN_LABEL = BASE_DIR / "yolo_dataset" / "labels" / "train"
YOLO_VAL_IMG = BASE_DIR / "yolo_dataset" / "images" / "val"
YOLO_VAL_LABEL = BASE_DIR / "yolo_dataset" / "labels" / "val"

# Créer les dossiers de validation s'ils n'existent pas
YOLO_VAL_IMG.mkdir(parents=True, exist_ok=True)
YOLO_VAL_LABEL.mkdir(parents=True, exist_ok=True)

# Nombre d'images par catégorie
N_PER_CLASS = 3

def get_image_basename(image_path):
    """Retourne le nom de base sans extension"""
    return Path(image_path).stem

def move_image_and_label(image_path, category):
    """Déplace une image et son label YOLO correspondant"""
    image_name = image_path.name
    base_name = get_image_basename(image_path)
    
    # Destination dans nos_dataset/Test
    dest_image = TEST_DIR / image_name
    
    # Trouver le label YOLO correspondant
    yolo_label = YOLO_TRAIN_LABEL / f"{base_name}.txt"
    yolo_image = YOLO_TRAIN_IMG / image_name
    
    print(f"  📦 {category}: {image_name}")
    
    # Déplacer l'image source
    if image_path.exists():
        shutil.move(str(image_path), str(dest_image))
        print(f"     ✓ Image déplacée: {image_path} → {dest_image}")
    else:
        print(f"     ⚠️  Image source introuvable: {image_path}")
    
    # Déplacer l'image YOLO
    if yolo_image.exists():
        dest_yolo_image = YOLO_VAL_IMG / image_name
        shutil.move(str(yolo_image), str(dest_yolo_image))
        print(f"     ✓ Image YOLO déplacée vers validation")
    else:
        print(f"     ⚠️  Image YOLO introuvable: {yolo_image}")
    
    # Déplacer le label YOLO
    if yolo_label.exists():
        dest_yolo_label = YOLO_VAL_LABEL / f"{base_name}.txt"
        shutil.move(str(yolo_label), str(dest_yolo_label))
        print(f"     ✓ Label YOLO déplacé vers validation")
    else:
        print(f"     ⚠️  Label YOLO introuvable: {yolo_label}")

def main():
    print("🔀 Déplacement aléatoire d'images vers le dataset de test")
    print("=" * 60)
    print()
    
    # Vérifier que le dossier d'entraînement existe
    if not TRAIN_DIR.exists():
        print(f"❌ Erreur: Dossier d'entraînement introuvable: {TRAIN_DIR}")
        return
    
    # Parcourir chaque catégorie
    categories = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    
    if not categories:
        print(f"❌ Aucune catégorie trouvée dans {TRAIN_DIR}")
        return
    
    total_moved = 0
    
    for category_dir in sorted(categories):
        category = category_dir.name
        
        # Lister toutes les images de cette catégorie
        images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.JPG"))
        
        if len(images) < N_PER_CLASS:
            print(f"⚠️  {category}: Seulement {len(images)} images disponibles (< {N_PER_CLASS})")
            n_to_move = len(images)
        else:
            n_to_move = N_PER_CLASS
        
        # Sélectionner aléatoirement
        selected = random.sample(images, n_to_move)
        
        print(f"\n📂 {category} ({n_to_move}/{len(images)} images):")
        
        # Déplacer chaque image et son label
        for image_path in selected:
            move_image_and_label(image_path, category)
            total_moved += 1
    
    print()
    print("=" * 60)
    print(f"✅ Terminé ! {total_moved} images déplacées vers le dataset de test")
    print()
    print("📊 Résumé:")
    print(f"   - Images source: {TRAIN_DIR}")
    print(f"   - Destination: {TEST_DIR}")
    print(f"   - Images YOLO: {YOLO_TRAIN_IMG} → {YOLO_VAL_IMG}")
    print(f"   - Labels YOLO: {YOLO_TRAIN_LABEL} → {YOLO_VAL_LABEL}")
    
    # Compter les images restantes
    print()
    print("📈 Images restantes par catégorie:")
    for category_dir in sorted(categories):
        category = category_dir.name
        remaining = len(list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.JPG")))
        print(f"   - {category}: {remaining} images")

if __name__ == "__main__":
    main()
