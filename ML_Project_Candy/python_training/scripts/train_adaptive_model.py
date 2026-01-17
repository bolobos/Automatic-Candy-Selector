#!/usr/bin/env python3
"""
Entraînement du modèle YOLOv8 adaptatif sur le dataset annoté
"""

from ultralytics import YOLO
from pathlib import Path
import sys

def train_adaptive_model():
    print("="*60)
    print("🚀 ENTRAÎNEMENT MODÈLE YOLO ADAPTATIF")
    print("="*60)
    print()
    
    project_root = Path.home() / "candy_project"
    dataset_yaml = project_root / "data" / "yolo_dataset" / "dataset.yaml"
    
    if not dataset_yaml.exists():
        print(f"❌ Dataset introuvable: {dataset_yaml}")
        sys.exit(1)
    
    print(f"📂 Dataset: {dataset_yaml}")
    print(f"🔄 Chargement YOLOv8n...")
    
    model = YOLO('yolov8n.pt')
    
    print(f"\n🏋️ Entraînement en cours...")
    print(f"   Epochs: 30 (adaptatif)")
    print(f"   Batch: 16")
    print(f"   Image size: 640")
    print()
    
    results = model.train(
        data=str(dataset_yaml),
        epochs=30,
        imgsz=640,
        batch=16,
        name='candy_adaptive',
        patience=5,
        save=True,
        exist_ok=True,
        verbose=True,
        plots=True
    )
    
    print(f"\n{'='*60}")
    print(f"✅ ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    print(f"📁 Modèle sauvegardé: runs/detect/candy_adaptive/weights/best.pt")
    print(f"{'='*60}\n")
    
    return results

if __name__ == "__main__":
    train_adaptive_model()
