from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cv2

# Configuration
test_images = Path(r"C:\Users\remic\Documents\Automatic-Candy-Selector\ML_Project_Candy\python_training\datasets\yolo_dataset\test\images")
test_labels = Path(r"C:\Users\remic\Documents\Automatic-Candy-Selector\ML_Project_Candy\python_training\datasets\yolo_dataset\test\labels")
model_path = r"C:\Users\remic\Documents\Automatic-Candy-Selector\runs\detect\candy_detector_yolov8m4\weights\best.pt"

# Charger le modèle
print("🔄 Chargement du modèle...")
model = YOLO(model_path)
print(f"✅ Modèle chargé\n")

# Lister les images
images = list(test_images.glob("*.jpg")) + list(test_images.glob("*.png"))
print(f"📸 {len(images)} images à analyser\n")

def calculate_iou(box1, box2):
    """Calcule l'IoU entre deux boxes (x_center, y_center, width, height)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convertir en coordonnées coin
    box1_x1, box1_y1 = x1 - w1/2, y1 - h1/2
    box1_x2, box1_y2 = x1 + w1/2, y1 + h1/2
    box2_x1, box2_y1 = x2 - w2/2, y2 - h2/2
    box2_x2, box2_y2 = x2 + w2/2, y2 + h2/2
    
    # Intersection
    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# Compteurs
labels_ok = 0
labels_invalides = 0
total_gt = 0
total_pred = 0
ious = []

# Traiter chaque image
for i, img_path in enumerate(images, 1):
    if i % 50 == 0:
        print(f"  Traité: {i}/{len(images)}")
    
    label_path = test_labels / (img_path.stem + ".txt")
    
    # Vérifier label
    if not label_path.exists() or label_path.stat().st_size == 0:
        labels_invalides += 1
        continue
    
    # Lire ground truth
    gt_boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                        gt_boxes.append((cls, x, y, w, h))
    except:
        labels_invalides += 1
        continue
    
    if not gt_boxes:
        labels_invalides += 1
        continue
    
    labels_ok += 1
    total_gt += len(gt_boxes)
    
    # Prédire
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    h_img, w_img = img.shape[:2]
    results = model(img, verbose=False)
    
    # Extraire prédictions
    pred_boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            
            # Normaliser
            x_c = ((x1 + x2) / 2) / w_img
            y_c = ((y1 + y2) / 2) / h_img
            w = (x2 - x1) / w_img
            h = (y2 - y1) / h_img
            
            pred_boxes.append((cls, x_c, y_c, w, h))
    
    total_pred += len(pred_boxes)
    
    # Matcher GT avec prédictions
    matched_gt = set()
    for pred in pred_boxes:
        pred_cls, pred_x, pred_y, pred_w, pred_h = pred
        best_iou = 0
        best_idx = -1
        
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            gt_cls, gt_x, gt_y, gt_w, gt_h = gt
            
            if pred_cls == gt_cls:
                iou = calculate_iou((pred_x, pred_y, pred_w, pred_h), 
                                   (gt_x, gt_y, gt_w, gt_h))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
        
        if best_iou > 0.5:
            matched_gt.add(best_idx)
            ious.append(best_iou)

# Résultats
print("\n" + "="*60)
print("📊 RÉSULTATS")
print("="*60)

# Labels
print(f"\n✅ Labels valides: {labels_ok}/{len(images)} ({labels_ok/len(images)*100:.1f}%)")
print(f"❌ Labels invalides: {labels_invalides}/{len(images)}")

# Prédictions
if total_gt > 0:
    detection_rate = len(ious) / total_gt * 100
    print(f"\n🎯 Détection: {len(ious)}/{total_gt} boxes détectées ({detection_rate:.1f}%)")
    print(f"📦 Boxes prédites: {total_pred}")
    
    if ious:
        avg_iou = np.mean(ious) * 100
        print(f"\n📏 Précision moyenne des bounding boxes: {avg_iou:.1f}%")
        print(f"   IoU min: {np.min(ious)*100:.1f}% | max: {np.max(ious)*100:.1f}%")
        
        # Résumé simplifié
        print("\n" + "="*60)
        if labels_ok == len(images) and detection_rate >= 95 and avg_iou >= 70:
            print("✅ EXCELLENT: Labels OK et prédictions précises!")
        elif labels_ok >= len(images)*0.9 and detection_rate >= 80 and avg_iou >= 60:
            print("👍 BON: Dataset utilisable")
        else:
            print("⚠️  À AMÉLIORER: Vérifier labels ou réentraîner")
        print("="*60)
