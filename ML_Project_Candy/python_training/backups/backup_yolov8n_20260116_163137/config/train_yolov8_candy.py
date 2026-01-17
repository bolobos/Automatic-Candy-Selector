#!/usr/bin/env python3
"""
Script d'entraînement YOLOv8 pour la détection de bonbons
Convertit automatiquement le dataset existant et entraîne le modèle
"""

import os
import shutil
from pathlib import Path
from ultralytics import YOLO

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
    (dest_base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dest_base / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Lire les classes depuis label_names.txt
    classes = []
    if Path("label_names.txt").exists():
        with open("label_names.txt", "r") as f:
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
    with open("yolo_models/candy.names", "w") as f:
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

def train_yolov8(model_size='n', epochs=100, img_size=640):
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
    if not Path("candy.yaml").exists():
        print("❌ Fichier candy.yaml introuvable!")
        print("💡 Créez-le avec la commande: python train_yolov8_candy.py --setup")
        return
    
    # Charger le modèle pré-entraîné
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Entraîner
    results = model.train(
        data='candy.yaml',
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        name='candy_detector',
        patience=50,  # Early stopping après 50 epochs sans amélioration
        save=True,
        plots=True,  # Générer des graphiques de performance
        verbose=True
    )
    
    print("\n✅ Entraînement terminé!")
    print(f"📊 Résultats dans: runs/detect/candy_detector/")
    
    # Exporter en ONNX pour Java
    print("\n📦 Export du modèle en ONNX pour Java...")
    best_model_path = "runs/detect/candy_detector/weights/best.pt"
    
    if Path(best_model_path).exists():
        model = YOLO(best_model_path)
        model.export(format='onnx', dynamic=False, simplify=True)
        
        # Déplacer le fichier ONNX dans yolo_models
        onnx_source = best_model_path.replace('.pt', '.onnx')
        onnx_dest = "yolo_models/candy_yolov8.onnx"
        
        if Path(onnx_source).exists():
            shutil.copy(onnx_source, onnx_dest)
            print(f"✅ Modèle ONNX copié vers: {onnx_dest}")
            print(f"\n🎯 Modifiez CandyDetectorYOLO.java pour utiliser:")
            print(f'   String modelPath = "{onnx_dest}";')
            print(f'   String classNamesFile = "yolo_models/candy.names";')
        else:
            print("❌ Erreur: fichier ONNX non trouvé")
    else:
        print("❌ Modèle entraîné introuvable")

def download_pretrained():
    """Télécharge un modèle YOLOv8 pré-entraîné COCO et l'exporte en ONNX"""
    print("📥 Téléchargement de YOLOv8n pré-entraîné...")
    
    # Créer le dossier si nécessaire
    Path("yolo_models").mkdir(exist_ok=True)
    
    # Télécharger et exporter
    model = YOLO('yolov8n.pt')
    print("📦 Export en ONNX...")
    model.export(format='onnx', dynamic=False, simplify=True)
    
    # Déplacer le fichier
    shutil.move('yolov8n.onnx', 'yolo_models/yolov8n.onnx')
    
    # Télécharger coco.names
    import urllib.request
    print("📋 Téléchargement de coco.names...")
    url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
    urllib.request.urlretrieve(url, "yolo_models/coco.names")
    
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
