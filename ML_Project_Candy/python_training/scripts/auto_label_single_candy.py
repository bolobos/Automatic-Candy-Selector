#!/usr/bin/env python3
"""
Auto-labellisation pour dataset avec UN bonbon par image
Utilise la détection de contours OpenCV pour générer les bounding boxes YOLO automatiquement
"""

import cv2
import numpy as np
from pathlib import Path
import os

def detect_candy_bbox(image_path, debug=False):
    """
    Détecte automatiquement le bonbon dans l'image et retourne la bounding box
    
    Returns:
        tuple: (x_center, y_center, width, height) normalisés entre 0 et 1
               ou None si aucun bonbon détecté
    """
    # Charger l'image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Impossible de lire {image_path}")
        return None
    
    height, width = img.shape[:2]
    
    # Convertir en HSV pour mieux détecter les couleurs vives des bonbons
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Créer un masque pour détecter les objets colorés (pas le fond blanc/gris)
    # Ajustez ces valeurs selon votre fond
    lower_bound = np.array([0, 30, 30])    # Évite le blanc/gris/noir
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Nettoyer le masque avec morphologie
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Trouver les contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"⚠️  Aucun contour trouvé dans {image_path.name}")
        return None
    
    # Prendre le plus grand contour (supposé être le bonbon)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculer la bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Ajouter une petite marge (5%)
    margin = 0.05
    x = max(0, int(x - w * margin))
    y = max(0, int(y - h * margin))
    w = min(width - x, int(w * (1 + 2 * margin)))
    h = min(height - y, int(h * (1 + 2 * margin)))
    
    # Normaliser pour YOLO (valeurs entre 0 et 1)
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    box_width = w / width
    box_height = h / height
    
    # Debug: afficher l'image avec la bbox
    if debug:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Detection", debug_img)
        cv2.waitKey(0)
    
    return (x_center, y_center, box_width, box_height)

def auto_label_dataset(source_dir="nos_dataset/Entrainement", output_dir="yolo_dataset", debug=False):
    """
    Auto-labellise tout le dataset
    
    Args:
        source_dir: Dossier source avec structure ClasseX/*.jpg
        output_dir: Dossier de sortie YOLO
        debug: Si True, affiche chaque détection pour vérification
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Créer la structure YOLO
    (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
    # Lire les classes depuis label_names.txt ou détecter automatiquement
    if Path("../datasets/label_names.txt").exists():
        with open("../datasets/label_names.txt", "r") as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    
    print(f"🏷️  Classes détectées: {classes}")
    
    # Créer candy.names
    with open("yolo_models/candy.names", "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    total_images = 0
    successful = 0
    failed = 0
    
    # Parcourir chaque classe
    for class_idx, class_name in enumerate(classes):
        class_dir = source_path / class_name
        
        if not class_dir.is_dir():
            continue
        
        print(f"\n📂 Traitement de {class_name} (classe {class_idx})...")
        
        # Parcourir chaque image
        for img_file in class_dir.glob("*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            total_images += 1
            
            # Détecter la bounding box
            bbox = detect_candy_bbox(img_file, debug=debug)
            
            if bbox is None:
                print(f"   ⚠️  Échec: {img_file.name}")
                failed += 1
                continue
            
            # Copier l'image
            dest_img = output_path / "images" / "train" / f"{class_name}_{img_file.name}"
            import shutil
            shutil.copy(img_file, dest_img)
            
            # Créer le fichier label YOLO
            label_file = output_path / "labels" / "train" / f"{class_name}_{img_file.stem}.txt"
            with open(label_file, "w") as f:
                # Format YOLO: class_id x_center y_center width height
                f.write(f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
            successful += 1
    
    print(f"\n{'='*50}")
    print(f"✅ Auto-labellisation terminée!")
    print(f"📊 Statistiques:")
    print(f"   - Total d'images: {total_images}")
    print(f"   - Succès: {successful}")
    print(f"   - Échecs: {failed}")
    print(f"   - Taux de réussite: {successful/total_images*100:.1f}%")
    print(f"\n📁 Dataset YOLO créé dans: {output_path}")
    print(f"🎯 Vous pouvez maintenant lancer:")
    print(f"   python train_yolov8_candy.py --train")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-labellisation pour bonbons')
    parser.add_argument('--source', type=str, default='nos_dataset/Entrainement',
                       help='Dossier source avec les images')
    parser.add_argument('--output', type=str, default='yolo_dataset',
                       help='Dossier de sortie YOLO')
    parser.add_argument('--debug', action='store_true',
                       help='Afficher chaque détection pour vérification')
    
    args = parser.parse_args()
    
    print("🍬 Auto-labellisation automatique pour dataset de bonbons\n")
    print("⚙️  Cette méthode fonctionne si:")
    print("   ✓ Vous avez UN bonbon par image")
    print("   ✓ Le bonbon est l'objet principal (plus grand contour)")
    print("   ✓ Le fond est relativement uniforme\n")
    
    auto_label_dataset(args.source, args.output, args.debug)
    
    print("\n💡 Conseil: Vérifiez quelques images dans yolo_dataset/ pour confirmer")
    print("   que les bounding boxes sont correctes avant d'entraîner!")
