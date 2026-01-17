#!/usr/bin/env python3
"""Export rapide YOLOv8 vers ONNX"""

print("🔄 Chargement des bibliothèques (2-3 min)...")
from ultralytics import YOLO
import shutil
from pathlib import Path

print("✅ Bibliothèques chargées!")
print("📦 Export du modèle en ONNX...")

# Charger le modèle
model = YOLO("yolo_training/candy_detector/weights/best.pt")

# Exporter en ONNX
model.export(format="onnx", imgsz=640)

# Copier vers yolo_models
onnx_path = Path("yolo_training/candy_detector/weights/best.onnx")
target_path = Path("yolo_models/candy_yolov8.onnx")
target_path.parent.mkdir(exist_ok=True)

if onnx_path.exists():
    shutil.copy(onnx_path, target_path)
    print(f"✅ Modèle exporté: {target_path}")
    print(f"📊 Taille: {target_path.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print(f"❌ Erreur: {onnx_path} introuvable")
