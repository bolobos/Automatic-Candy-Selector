#!/usr/bin/env python3
"""
Export ONNX et affichage des statistiques d'entraînement
"""
from ultralytics import YOLO
from pathlib import Path
import shutil
import csv
import os

# Remonter au dossier racine du projet
os.chdir('/mnt/c/Users/remic/Documents/Automatic-Candy-Selector')

# Charger le meilleur modèle
best_model = Path('runs/detect/candy_detector7/weights/best.pt')
print(f'📦 Export ONNX du modèle: {best_model}')

model = YOLO(str(best_model))
export_path = model.export(format='onnx', dynamic=False, simplify=True)

# Copier vers yolo_models
onnx_dest = Path('ML_Project_Candy/python_training/yolo_models/candy_yolov8.onnx')
onnx_dest.parent.mkdir(exist_ok=True, parents=True)

if Path(export_path).exists():
    shutil.copy(export_path, onnx_dest)
    print(f'✅ Modèle ONNX exporté: {onnx_dest}')
    print(f'📦 Taille: {onnx_dest.stat().st_size / 1024 / 1024:.1f} MB')

# Afficher les stats
csv_path = Path('runs/detect/candy_detector7/results.csv')
if csv_path.exists():
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    last = rows[-1]
    
    # Trouver la meilleure epoch par mAP50
    best_map = max(rows, key=lambda x: float(x['metrics/mAP50(B)']))
    
    total_time = float(last['time'])
    
    print()
    print('='*60)
    print('📊 STATISTIQUES ENTRAÎNEMENT YOLOv8s')
    print('='*60)
    print(f'⏱️  Temps total: {total_time:.1f}s ({total_time/60:.1f}min / {total_time/3600:.2f}h)')
    print(f'🔄 Epochs complétés: {int(float(last["epoch"]))}')
    print(f'⏱️  Temps moyen/epoch: {total_time/int(float(last["epoch"])):.1f}s')
    print()
    print(f'🏆 MEILLEURE EPOCH ({int(float(best_map["epoch"]))}):')
    print(f'   mAP50: {float(best_map["metrics/mAP50(B)"]):.4f}')
    print(f'   mAP50-95: {float(best_map["metrics/mAP50-95(B)"]):.4f}')
    print(f'   Precision: {float(best_map["metrics/precision(B)"]):.4f}')
    print(f'   Recall: {float(best_map["metrics/recall(B)"]):.4f}')
    print()
    print(f'📈 DERNIÈRE EPOCH ({int(float(last["epoch"]))}):')
    print(f'   mAP50: {float(last["metrics/mAP50(B)"]):.4f}')
    print(f'   mAP50-95: {float(last["metrics/mAP50-95(B)"]):.4f}')
    print(f'   Precision: {float(last["metrics/precision(B)"]):.4f}')
    print(f'   Recall: {float(last["metrics/recall(B)"]):.4f}')
    print(f'   Box Loss: {float(last["train/box_loss"]):.4f}')
    print(f'   Cls Loss: {float(last["train/cls_loss"]):.4f}')
    print('='*60)
    print()
    print('📂 Dataset:')
    print(f'   Train: 1294 images (85%)')
    print(f'   Val: 228 images (15%)')
    print(f'   Classes: 6 (Croco, Dragibus, Oeuf, Ourson, Schtroumpf)')
    print('='*60)
