#!/usr/bin/env python3
"""
Script d'entraînement YOLOv8 pour la détection de bonbons
Convertit automatiquement le dataset existant et entraîne le modèle
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# Répertoire du script (pour chemins relatifs)
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

def prepare_dataset():
    """
    Prépare le dataset en convertissant la structure existante
    nos_dataset/Entrainement/ClasseX/*.jpg -> yolo_dataset/images/train/*.jpg
    """
    print("📁 Préparation du dataset YOLO...")
    
    # Chemins source et destination
    source_dir = Path("nos_dataset/Entrainement")
    dest_base = Path("yolo_dataset")
    
    # Créer la structure YOLO
    (dest_base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dest_base / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dest_base / "images" / "test").mkdir(parents=True, exist_ok=True)
    (dest_base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dest_base / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (dest_base / "labels" / "test").mkdir(parents=True, exist_ok=True)
    
    # Lire les classes depuis label_names.txt
    classes = []
    label_names_file = PROJECT_DIR / "datasets" / "label_names.txt"
    if label_names_file.exists():
        with open(label_names_file, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
    else:
        # Détecter automatiquement depuis les dossiers
        if source_dir.exists():
            classes = [d.name for d in source_dir.iterdir() if d.is_dir()]
        else:
            print("❌ Dataset source introuvable!")
            return False
    
    print(f"🏷️  Classes détectées: {classes}")
    
    # Créer le fichier candy.names
    candy_names_file = PROJECT_DIR / "yolo_models" / "candy.names"
    with open(candy_names_file, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    print(f"✅ Fichier candy.names créé avec {len(classes)} classes")
    
    # IMPORTANT: Ce script créé uniquement la structure
    # Vous devez annoter vos images avec LabelImg ou Roboflow pour créer les labels
    print("\n⚠️  ATTENTION: Ce script ne créé que la structure du dataset.")
    print("📝 Vous devez annoter vos images avec:")
    print("   - LabelImg: https://github.com/heartexlabs/labelImg")
    print("   - Roboflow: https://roboflow.com (recommandé, automatique)")
    print("   - CVAT: https://cvat.org")
    print("\n💡 Roboflow peut auto-générer le format YOLO à partir de vos images!")
    
    return True

def train_yolov8(model_size='m', epochs=150, img_size=640):
    """
    Entraîne YOLOv8 sur le dataset de bonbons
    
    Args:
        model_size: 'n', 's', 'm', 'l', 'x' (nano à extra-large)
        epochs: Nombre d'époques d'entraînement
        img_size: Taille des images d'entrée
    """
    print(f"\n🚀 Entraînement YOLOv8{model_size} sur dataset de bonbons...")
    print(f"⚙️  Paramètres: {epochs} epochs, {img_size}px")
    
    # Vérifier si candy.yaml existe
    candy_yaml = PROJECT_DIR / "candy.yaml"
    if not candy_yaml.exists():
        print(f"❌ Fichier {candy_yaml} introuvable!")
        print("💡 Créez-le avec la commande: python train_yolov8_candy.py --setup")
        return
    
    # Charger le modèle pré-entraîné depuis yolo_models
    model_path = PROJECT_DIR / 'yolo_models' / f'yolov8{model_size}.pt'
    if not model_path.exists():
        print(f"⚠️  Modèle {model_path} introuvable, téléchargement automatique...")
        # Créer le dossier s'il n'existe pas
        model_path.parent.mkdir(exist_ok=True)
        # Télécharger avec YOLO, qui va le sauvegarder dans le dossier courant
        temp_model = YOLO(f'yolov8{model_size}.pt')
        # Déplacer vers yolo_models
        temp_path = Path(f'yolov8{model_size}.pt')
        if temp_path.exists():
            shutil.move(str(temp_path), str(model_path))
        model = temp_model
    else:
        print(f"📦 Chargement du modèle: {model_path}")
        model = YOLO(str(model_path))
    
    # Entraîner
    project_path = str(PROJECT_DIR.parent.parent / "runs" / "detect")
    name = f'candy_detector_yolov8{model_size}'
    
    results = model.train(
        data=str(candy_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        project=project_path,
        name=name,
        exist_ok=False,  # Ne pas écraser, créer candy_detector_yolov8m2, etc.
        patience=30,  # Early stopping après 30 epochs sans amélioration
        save_period=10,
        cache=True,
        workers=1,
        save=True,
        plots=True,  # Générer des graphiques de performance
        verbose=True,
        val=True,
        amp=True  # Mixed Precision pour réduire l'utilisation GPU
    )
    
    # Récupérer le vrai chemin créé par YOLO (avec numéro incrémenté si besoin)
    actual_save_dir = results.save_dir  # Chemin réel du dossier de sauvegarde
    
    print("\n✅ Entraînement terminé!")
    print(f"📊 Résultats dans: {actual_save_dir}")
    
    # Exporter en ONNX pour Java
    print("\n📦 Export du modèle en ONNX pour Java...")
    best_model_path = Path(actual_save_dir) / "weights" / "best.pt"
    
    if best_model_path.exists():
        model = YOLO(str(best_model_path))
        model.export(format='onnx', dynamic=False, simplify=True)
        
        # Déplacer le fichier ONNX dans le dossier models
        onnx_source = str(best_model_path).replace('.pt', '.onnx')
        models_dir = Path('../models')
        models_dir.mkdir(exist_ok=True)
        onnx_dest = models_dir / f'candy_yolov8{model_size}.onnx'
        
        if Path(onnx_source).exists():
            shutil.copy(onnx_source, str(onnx_dest))
            print(f"✅ Modèle ONNX exporté vers: {onnx_dest}")
            print(f"\n🎯 Pour Java, utilisez:")
            print(f'   String modelPath = "{onnx_dest}";')
            print(f'   String classNamesFile = "../python_training/datasets/label_names.txt";')
        else:
            print("❌ Erreur: fichier ONNX non trouvé")
    else:
        print("❌ Modèle entraîné introuvable")

def download_pretrained():
    """Télécharge un modèle YOLOv8 pré-entraîné COCO et l'exporte en ONNX"""
    print("📥 Téléchargement de YOLOv8n pré-entraîné...")
    
    # Créer le dossier si nécessaire
    yolo_models_dir = PROJECT_DIR / "yolo_models"
    yolo_models_dir.mkdir(exist_ok=True)
    
    # Télécharger et exporter
    model = YOLO('yolov8n.pt')
    print("📦 Export en ONNX...")
    model.export(format='onnx', dynamic=False, simplify=True)
    
    # Déplacer le fichier
    shutil.move('yolov8n.onnx', str(yolo_models_dir / 'yolov8n.onnx'))
    
    # Télécharger coco.names
    import urllib.request
    print("📋 Téléchargement de coco.names...")
    url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    urllib.request.urlretrieve(url, str(yolo_models_dir / 'coco.names'))
    
    print("✅ Modèle YOLOv8n ONNX prêt à utiliser!")
    print("🎯 Vous pouvez maintenant lancer CandyDetectorYOLO.java")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraîner YOLOv8 pour détection de bonbons')
    parser.add_argument('--setup', action='store_true', 
                       help='Préparer le dataset (structure uniquement)')
    parser.add_argument('--train', action='store_true', 
                       help='Entraîner le modèle sur vos données annotées')
    parser.add_argument('--download', action='store_true', 
                       help='Télécharger YOLOv8 pré-entraîné COCO')
    parser.add_argument('--model', type=str, default='n', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Taille du modèle (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Nombre d\'époques d\'entraînement')
    parser.add_argument('--img-size', type=int, default=640, 
                       help='Taille des images')
    
    args = parser.parse_args()
    
    if args.setup:
        prepare_dataset()
    elif args.train:
        train_yolov8(args.model, args.epochs, args.img_size)
    elif args.download:
        download_pretrained()
    else:
        print("🍬 Script d'entraînement YOLOv8 pour bonbons\n")
        print("Usage:")
        print("  python train_yolov8_candy.py --download        # Télécharger modèle pré-entraîné")
        print("  python train_yolov8_candy.py --setup           # Préparer le dataset")
        print("  python train_yolov8_candy.py --train           # Entraîner sur vos données")
        print("  python train_yolov8_candy.py --train --model s # Entraîner avec YOLOv8s")
        print("\n💡 Commencez par --download pour tester avec le modèle COCO générique")
