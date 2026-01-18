#!/usr/bin/env python3
"""Script pour exporter le modèle YOLOv8 en ONNX"""
from ultralytics import YOLO
import shutil
from pathlib import Path

# Répertoire du script
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

# Charger le meilleur modèle
model_path = PROJECT_DIR.parent.parent / "runs" / "detect" / "candy_detector_yolov8m4" / "weights" / "best.pt"
print(f"📥 Chargement du modèle: {model_path}")
model = YOLO(str(model_path))

# Exporter en ONNX
print("🔄 Export en ONNX...")
model.export(format='onnx', dynamic=False, simplify=True)

# Copier vers yolo_models
onnx_source = str(model_path).replace('.pt', '.onnx')
onnx_dest = PROJECT_DIR / "yolo_models" / "candy_yolov8.onnx"
shutil.copy(onnx_source, onnx_dest)

print(f"✅ Modèle ONNX exporté: {onnx_dest}")
print("🎯 Prêt à utiliser dans Java!")
