#!/usr/bin/env python3
"""
Script pour détecter automatiquement les bonbons dans un dossier d'images
et créer les fichiers labels au format YOLO
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
from tqdm import tqdm

# Répertoire du script
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Configuration
MODEL_PATH = "../../../runs/detect/candy_detector_yolov8m4/weights/best.pt"
IMAGES_FOLDER = "/mnt/c/Users/remic/Documents/Automatic-Candy-Selector/PXL_20260118_100006771_frames"
CONFIDENCE_THRESHOLD = 0.25  # Seuil de confiance (ajustable)

# Classes de bonbons (dans l'ordre du modèle)
CLASSES = ["Tagada", "Dragibus", "Ourson", "Oeuf", "Croco", "Schtroumpf"]

def create_yolo_labels(image_folder, model_path, conf_threshold=0.25):
    """
    Détecte les bonbons dans toutes les images et crée les fichiers labels
    """
    # Charger le modèle
    print(f"🔄 Chargement du modèle: {model_path}")
    model = YOLO(model_path)
    
    # Lister toutes les images
    image_folder = Path(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in image_folder.glob('*') if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"❌ Aucune image trouvée dans {image_folder}")
        return
    
    print(f"📸 {len(images)} images à traiter")
    print(f"🎯 Seuil de confiance: {conf_threshold}")
    print()
    
    stats = {
        'total_images': len(images),
        'images_with_detections': 0,
        'total_detections': 0,
        'detections_by_class': {cls: 0 for cls in CLASSES}
    }
    
    # Traiter chaque image
    for image_path in tqdm(images, desc="🍬 Détection"):
        # Prédiction
        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            verbose=False
        )[0]
        
        # Créer le fichier label
        label_path = image_path.with_suffix('.txt')
        
        # Obtenir les dimensions de l'image
        img = cv2.imread(str(image_path))
        img_height, img_width = img.shape[:2]
        
        detections = []
        
        # Extraire les détections
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Récupérer les coordonnées (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Convertir en format YOLO (x_center, y_center, width, height) normalisé
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                detections.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Statistiques
                stats['total_detections'] += 1
                if class_id < len(CLASSES):
                    stats['detections_by_class'][CLASSES[class_id]] += 1
        
        # Écrire le fichier label
        with open(label_path, 'w') as f:
            if detections:
                f.write('\n'.join(detections) + '\n')
                stats['images_with_detections'] += 1
            # Si pas de détections, créer un fichier vide
    
    # Afficher les statistiques
    print()
    print("=" * 60)
    print("📊 STATISTIQUES DE DÉTECTION")
    print("=" * 60)
    print(f"Images traitées:          {stats['total_images']}")
    print(f"Images avec détections:   {stats['images_with_detections']} ({stats['images_with_detections']/stats['total_images']*100:.1f}%)")
    print(f"Total de détections:      {stats['total_detections']}")
    print()
    print("Détections par classe:")
    for cls, count in stats['detections_by_class'].items():
        if count > 0:
            print(f"  {cls:15s}: {count:4d}")
    print("=" * 60)
    print()
    print(f"✅ Fichiers labels créés dans: {image_folder}")
    print(f"📝 Format: classe x_center y_center width height (normalisé 0-1)")
    print()
    print("💡 Prochaine étape: Vérifier/corriger manuellement avec un outil d'annotation")

if __name__ == "__main__":
    create_yolo_labels(IMAGES_FOLDER, MODEL_PATH, CONFIDENCE_THRESHOLD)
