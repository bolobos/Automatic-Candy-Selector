import os
import shutil
import random
from pathlib import Path
from datetime import datetime

def shuffle_dataset(dataset_path, train_ratio=0.85, seed=None):
    """
    Redistribue aléatoirement les images et labels entre train et validation
    en gardant le même ratio.
    
    Args:
        dataset_path: Chemin vers yolo_dataset
        train_ratio: Ratio d'images pour l'entraînement (par défaut 0.85)
        seed: Graine aléatoire pour reproductibilité (None = aléatoire)
    """
    
    if seed is not None:
        random.seed(seed)
    
    train_images_dir = Path(dataset_path) / "train" / "images"
    train_labels_dir = Path(dataset_path) / "train" / "labels"
    val_images_dir = Path(dataset_path) / "val" / "images"
    val_labels_dir = Path(dataset_path) / "val" / "labels"
    
    print("=== Collecte de toutes les images et labels ===\n")
    
    # Collecter toutes les paires image-label
    all_pairs = []
    
    # Depuis train
    for img_file in train_images_dir.glob("*.*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            label_file = train_labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                all_pairs.append({
                    'image': img_file,
                    'label': label_file,
                    'source': 'train'
                })
            else:
                print(f"⚠️ Image sans label: {img_file.name}")
    
    # Depuis validation
    for img_file in val_images_dir.glob("*.*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            label_file = val_labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                all_pairs.append({
                    'image': img_file,
                    'label': label_file,
                    'source': 'val'
                })
            else:
                print(f"⚠️ Image sans label: {img_file.name}")
    
    total_pairs = len(all_pairs)
    print(f"\n✓ Total paires image-label trouvées: {total_pairs}")
    
    # Mélanger aléatoirement
    random.shuffle(all_pairs)
    print("✓ Paires mélangées aléatoirement")
    
    # Calculer la nouvelle répartition
    train_count = int(total_pairs * train_ratio)
    val_count = total_pairs - train_count
    
    # Vérifier qu'on a trouvé des paires
    if total_pairs == 0:
        print("\n❌ Aucune paire image-label trouvée!")
        print(f"Vérifiez que le chemin est correct: {dataset_path}")
        return
    
    print(f"\n=== Nouvelle répartition ===")
    print(f"Train: {train_count} images ({train_count/total_pairs*100:.1f}%)")
    print(f"Val:   {val_count} images ({val_count/total_pairs*100:.1f}%)")
    
    # Confirmation
    response = input("\nConfirmer la redistribution? (oui/non): ")
    if response.lower() not in ['oui', 'o', 'yes', 'y']:
        print("Annulé.")
        return
    
    # Créer une sauvegarde
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== Création de la sauvegarde {backup_time} ===")
    
    # Supprimer les caches qui seront invalides
    for cache_file in [train_labels_dir.parent / "labels.cache", 
                       val_labels_dir.parent / "labels.cache"]:
        if cache_file.exists():
            cache_file.unlink()
            print(f"✓ Supprimé: {cache_file.name}")
    
    # Déplacer les fichiers
    print(f"\n=== Redistribution des fichiers ===")
    moved = 0
    errors = []
    
    for i, pair in enumerate(all_pairs):
        # Déterminer la destination
        if i < train_count:
            dest_images = train_images_dir
            dest_labels = train_labels_dir
            dest_name = "train"
        else:
            dest_images = val_images_dir
            dest_labels = val_labels_dir
            dest_name = "val"
        
        # Si déjà dans le bon dossier, ne rien faire
        current_dir = "train" if pair['source'] == 'train' else "val"
        if current_dir == dest_name:
            continue
        
        try:
            # Déplacer l'image
            new_img_path = dest_images / pair['image'].name
            shutil.move(str(pair['image']), str(new_img_path))
            
            # Déplacer le label
            new_label_path = dest_labels / pair['label'].name
            shutil.move(str(pair['label']), str(new_label_path))
            
            moved += 1
            if moved % 50 == 0:
                print(f"  {moved} paires déplacées...")
                
        except Exception as e:
            errors.append(f"{pair['image'].name}: {str(e)}")
            print(f"✗ Erreur: {pair['image'].name}")
    
    print(f"\n=== Résultat ===")
    print(f"✓ {moved} paires déplacées")
    
    if errors:
        print(f"\n⚠️ {len(errors)} erreurs:")
        for error in errors[:10]:  # Afficher max 10 erreurs
            print(f"  - {error}")
    
    # Vérification finale
    final_train = len(list(train_images_dir.glob("*.*")))
    final_val = len(list(val_images_dir.glob("*.*")))
    
    print(f"\n=== Distribution finale ===")
    print(f"Train: {final_train} images")
    print(f"Val:   {final_val} images")
    print(f"Total: {final_train + final_val} images")
    print("\n✓ Redistribution terminée!")


if __name__ == "__main__":
    # Utiliser un chemin relatif depuis le script
    script_dir = Path(__file__).parent
    dataset_path = script_dir / "datasets" / "yolo_dataset"
    
    # Utiliser None pour un mélange vraiment aléatoire, ou un nombre pour reproductibilité
    shuffle_dataset(dataset_path, train_ratio=0.85, seed=None)
