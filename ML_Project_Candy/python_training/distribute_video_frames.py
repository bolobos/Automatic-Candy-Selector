#!/usr/bin/env python3
"""
Script pour répartir les images et labels d'une vidéo dans train/val/test
en respectant l'ordre séquentiel des frames
"""

from pathlib import Path
import shutil
import random

# Configuration
SOURCE_DIR = Path("/mnt/c/Users/remic/Documents/Automatic-Candy-Selector/PXL_20260118_100006771_frames")
DEST_DIR = Path("datasets/yolo_dataset")

# Ratios (doivent totaliser 100)
TRAIN_RATIO = 85
VAL_RATIO = 10
TEST_RATIO = 5

def distribute_video_frames():
    """
    Répartit les frames de vidéo dans train/val/test en respectant l'ordre séquentiel
    Pour éviter la fuite de données, on divise en blocs plutôt que de mélanger aléatoirement
    """
    
    # Les images et labels sont dans des sous-dossiers
    images_dir = SOURCE_DIR / 'images'
    labels_dir = SOURCE_DIR / 'labels'
    
    # Lister toutes les paires image/label
    image_files = sorted(images_dir.glob('*.jpg'))
    
    pairs = []
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"⚠️  Pas de label pour: {img_path.name}")
    
    if not pairs:
        print("❌ Aucune paire image/label trouvée!")
        return
    
    total = len(pairs)
    print(f"\n📊 Total de paires image/label: {total}")
    print(f"📁 Source: {SOURCE_DIR}")
    print(f"📁 Destination: {DEST_DIR}")
    print()
    
    # Calculer les tailles de chaque set
    train_size = int(total * TRAIN_RATIO / 100)
    val_size = int(total * VAL_RATIO / 100)
    test_size = total - train_size - val_size  # Le reste va dans test
    
    print(f"🎯 Répartition:")
    print(f"   Train: {train_size} images ({train_size/total*100:.1f}%)")
    print(f"   Val:   {val_size} images ({val_size/total*100:.1f}%)")
    print(f"   Test:  {test_size} images ({test_size/total*100:.1f}%)")
    print()
    
    # Créer une liste d'indices et mélanger les GROUPES (pas les images individuelles)
    # Pour respecter l'ordre séquentiel, on divise la vidéo en segments
    # et on assigne chaque segment à un set
    
    # Stratégie: diviser en petits blocs de ~10 frames et répartir ces blocs
    block_size = 10
    blocks = []
    for i in range(0, total, block_size):
        blocks.append(list(range(i, min(i + block_size, total))))
    
    # Mélanger les blocs
    random.shuffle(blocks)
    
    # Aplatir les blocs pour obtenir les indices
    indices = []
    for block in blocks:
        indices.extend(block)
    
    # Diviser les indices selon les ratios
    train_indices = set(indices[:train_size])
    val_indices = set(indices[train_size:train_size + val_size])
    test_indices = set(indices[train_size + val_size:])
    
    # Créer les dossiers de destination
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            dest = DEST_DIR / split / subdir
            dest.mkdir(parents=True, exist_ok=True)
    
    # Copier les fichiers
    print("📦 Copie des fichiers...")
    
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for idx, (img_path, label_path) in enumerate(pairs):
        # Déterminer le split
        if idx in train_indices:
            split = 'train'
        elif idx in val_indices:
            split = 'val'
        else:
            split = 'test'
        
        # Copier l'image
        dest_img = DEST_DIR / split / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # Copier le label
        dest_label = DEST_DIR / split / 'labels' / label_path.name
        shutil.copy2(label_path, dest_label)
        
        stats[split] += 1
    
    print("✅ Copie terminée!")
    print()
    print("=" * 60)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"Train: {stats['train']} images dans {DEST_DIR}/train/")
    print(f"Val:   {stats['val']} images dans {DEST_DIR}/val/")
    print(f"Test:  {stats['test']} images dans {DEST_DIR}/test/")
    print("=" * 60)
    print()
    print("💡 Les frames ont été réparties par blocs pour respecter la continuité vidéo")
    print("   et éviter la fuite de données entre les ensembles.")

if __name__ == "__main__":
    random.seed(42)  # Pour reproductibilité
    distribute_video_frames()
