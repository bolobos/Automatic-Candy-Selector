#!/usr/bin/env python3
"""
Test du modèle YOLOv8 entraîné sur le dataset de test
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def test_model(model_path, test_dataset_path, output_dir, conf_threshold=0.25):
    """
    Teste le modèle YOLOv8 sur le dataset de test.
    
    Args:
        model_path: Chemin vers le modèle (.pt ou .onnx)
        test_dataset_path: Dossier du dataset de test
        output_dir: Dossier pour sauvegarder les prédictions
        conf_threshold: Seuil de confiance minimal
    """
    model_path = Path(model_path)
    test_dataset_path = Path(test_dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Test du modèle : {model_path}")
    print(f"📂 Dataset de test : {test_dataset_path}")
    print(f"💾 Résultats dans : {output_dir}\n")
    
    # Charger le modèle
    print("📥 Chargement du modèle...")
    model = YOLO(str(model_path))
    
    # Lire les noms de classes
    label_file = Path(__file__).parent / "label_names.txt"
    with open(label_file, 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Classes : {class_names}\n")
    
    # Statistiques
    stats = {
        'total': 0,
        'detected': 0,
        'by_class': {name: {'correct': 0, 'total': 0} for name in class_names}
    }
    
    # Parcourir chaque classe
    for class_name in class_names:
        class_path = test_dataset_path / class_name
        if not class_path.exists():
            print(f"⚠️  Dossier {class_name} introuvable")
            continue
        
        # Trouver toutes les images
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        print(f"🔍 Test de {class_name} : {len(image_files)} images")
        
        for img_path in image_files:
            stats['total'] += 1
            stats['by_class'][class_name]['total'] += 1
            
            # Faire la prédiction
            results = model(str(img_path), conf=conf_threshold, verbose=False)
            
            # Analyser les résultats
            img = cv2.imread(str(img_path))
            detected_class = None
            max_conf = 0
            
            for result in results:
                boxes = result.boxes
                if len(boxes) > 0:
                    stats['detected'] += 1
                    # Prendre la détection avec la plus haute confiance
                    best_box = boxes[boxes.conf.argmax()]
                    cls_id = int(best_box.cls[0])
                    confidence = float(best_box.conf[0])
                    
                    if confidence > max_conf:
                        max_conf = confidence
                        detected_class = class_names[cls_id]
                    
                    # Dessiner la bounding box
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    color = (0, 255, 0) if detected_class == class_name else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Ajouter le label
                    label = f"{detected_class} {confidence:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Vérifier si la prédiction est correcte
            if detected_class == class_name:
                stats['by_class'][class_name]['correct'] += 1
            
            # Sauvegarder l'image avec la prédiction
            output_class_dir = output_dir / class_name
            output_class_dir.mkdir(exist_ok=True)
            output_path = output_class_dir / img_path.name
            cv2.imwrite(str(output_path), img)
        
        # Afficher les stats pour cette classe
        correct = stats['by_class'][class_name]['correct']
        total = stats['by_class'][class_name]['total']
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"   ✅ {class_name}: {correct}/{total} ({accuracy:.1f}%)")
    
    # Afficher les statistiques globales
    print(f"\n{'='*60}")
    print("📊 Résultats globaux :")
    print(f"   - Images testées : {stats['total']}")
    print(f"   - Détections : {stats['detected']} ({stats['detected']/max(stats['total'],1)*100:.1f}%)")
    
    total_correct = sum(s['correct'] for s in stats['by_class'].values())
    global_accuracy = (total_correct / stats['total'] * 100) if stats['total'] > 0 else 0
    print(f"   - Précision globale : {total_correct}/{stats['total']} ({global_accuracy:.1f}%)")
    
    print(f"\n📁 Prédictions sauvegardées dans : {output_dir}")
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Test du modèle YOLOv8 sur le dataset de test')
    parser.add_argument('--model', type=str, 
                        default='yolo_training/candy_detector/weights/best.pt',
                        help='Chemin vers le modèle YOLOv8 (.pt ou .onnx)')
    parser.add_argument('--test-dir', type=str, 
                        default='nos_dataset/Test',
                        help='Dossier du dataset de test')
    parser.add_argument('--output', type=str, 
                        default='test_predictions',
                        help='Dossier pour sauvegarder les prédictions')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Seuil de confiance minimal (0-1)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model
    test_path = script_dir / args.test_dir
    output_path = script_dir / args.output
    
    print("🍬 Test du modèle YOLOv8 - Détection de bonbons")
    print("="*60 + "\n")
    
    # Vérifier que le modèle existe
    if not model_path.exists():
        print(f"❌ Modèle introuvable : {model_path}")
        print("   Lancer d'abord : python train_yolov8_candy.py --train")
        return
    
    # Vérifier que le dataset de test existe
    if not test_path.exists():
        print(f"❌ Dataset de test introuvable : {test_path}")
        return
    
    # Afficher la taille du modèle
    model_size = model_path.stat().st_size / 1024 / 1024
    print(f"📦 Taille du modèle : {model_size:.2f} MB")
    print(f"🎯 Seuil de confiance : {args.conf}\n")
    
    # Tester le modèle
    stats = test_model(model_path, test_path, output_path, args.conf)
    
    print("✅ Test terminé !")


if __name__ == '__main__':
    main()
