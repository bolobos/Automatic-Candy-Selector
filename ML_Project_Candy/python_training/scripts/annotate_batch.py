#!/usr/bin/env python3
"""
Script d'annotation automatique en mode batch (sans GUI)
Pour WSL - Version simplifiée du workflow adaptatif
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import shutil

class CandyBatchAnnotator:
    """Annotateur automatique en mode batch pour WSL"""
    
    def __init__(self, images_dir, base_model='yolov8n.pt', project_root=None):
        if project_root is None:
            project_root = Path.home() / "candy_project"
        
        self.project_root = Path(project_root)
        self.images_dir = Path(images_dir)
        self.annotations_dir = self.project_root / "data" / "annotations"
        self.yolo_dataset_dir = self.project_root / "data" / "yolo_dataset"
        self.backups_dir = self.project_root / "data" / "backups"
        self.models_dir = self.project_root / "models"
        
        for d in [self.annotations_dir, self.backups_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.image_files = self._load_image_files()
        self.total_images = len(self.image_files)
        
        if self.total_images == 0:
            raise ValueError(f"Aucune image trouvée dans {self.images_dir}")
        
        print(f"📸 {self.total_images} images détectées")
        
        self.annotations_file = self.annotations_dir / "annotations.json"
        self.annotations = self._load_annotations()
        
        pretrained_model = self.models_dir / "pretrained" / base_model
        if not pretrained_model.exists():
            pretrained_model = base_model
        
        print(f"🔄 Chargement modèle: {pretrained_model}")
        self.model = YOLO(pretrained_model)
        self.current_model_path = pretrained_model
    
    def _load_image_files(self):
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        for ext in extensions:
            files.extend(self.images_dir.glob(f"*{ext}"))
            files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        return sorted(files)
    
    def _load_annotations(self):
        if self.annotations_file.exists():
            with open(self.annotations_file, 'r') as f:
                annotations = json.load(f)
            print(f"📂 Annotations chargées: {len(annotations)} entrées")
            return annotations
        else:
            annotations = []
            for img_file in self.image_files:
                annotations.append({
                    'image': img_file.name,
                    'bbox': None,
                    'verified': False,
                    'confidence': 0.0
                })
            return annotations
    
    def annotate_all(self):
        """Pré-annote toutes les images automatiquement"""
        print(f"\n{'='*60}")
        print(f"🔍 ANNOTATION AUTOMATIQUE - {self.total_images} images")
        print(f"{'='*60}\n")
        
        processed = 0
        for idx, img_file in enumerate(self.image_files):
            if (idx + 1) % 100 == 0:
                print(f"  Progression: {idx+1}/{self.total_images} images...")
            
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"  ⚠️ Erreur lecture {img_file.name}")
                continue
            
            h, w = img.shape[:2]
            
            # Détection YOLO
            results = self.model(img, conf=0.25, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                max_idx = np.argmax(areas)
                
                best_box = boxes[max_idx]
                conf = float(results[0].boxes.conf[max_idx])
                
                bbox = [int(best_box[0]), int(best_box[1]), 
                       int(best_box[2]), int(best_box[3])]
            else:
                bbox = [w//4, h//4, 3*w//4, 3*h//4]
                conf = 0.0
            
            self.annotations[idx]['bbox'] = bbox
            self.annotations[idx]['confidence'] = conf
            self.annotations[idx]['verified'] = True
            processed += 1
        
        print(f"\n✅ Annotation terminée: {processed}/{self.total_images} images")
        self.save_annotations()
        
        # Créer dataset YOLO
        self.create_yolo_dataset()
    
    def create_yolo_dataset(self):
        """Crée le dataset formaté YOLO"""
        print(f"\n📦 Création dataset YOLO...")
        
        train_dir = self.yolo_dataset_dir / "train"
        train_images_dir = train_dir / "images"
        train_labels_dir = train_dir / "labels"
        
        for d in [train_images_dir, train_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        for ann in self.annotations:
            if not ann.get('bbox'):
                continue
            
            img_name = ann['image']
            bbox = ann['bbox']
            
            src_img = self.images_dir / img_name
            dst_img = train_images_dir / img_name
            shutil.copy2(src_img, dst_img)
            
            img = cv2.imread(str(src_img))
            h, w = img.shape[:2]
            
            x1, y1, x2, y2 = bbox
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            
            label_file = train_labels_dir / f"{img_name.rsplit('.', 1)[0]}.txt"
            with open(label_file, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Créer dataset.yaml
        yaml_file = self.yolo_dataset_dir / "dataset.yaml"
        yaml_content = f"""path: {self.yolo_dataset_dir}
train: train/images
val: train/images

nc: 1
names: ['candy']
"""
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Dataset YOLO créé: {yaml_file}")
    
    def save_annotations(self):
        """Sauvegarde annotations en JSON avec backup"""
        if self.annotations_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backups_dir / f"annotations_{timestamp}.json"
            shutil.copy2(self.annotations_file, backup_file)
        
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"💾 Annotations sauvegardées: {self.annotations_file}")


def main():
    project_root = Path.home() / "candy_project"
    images_dir = project_root / "data" / "raw_images"
    
    if not images_dir.exists() or not any(images_dir.iterdir()):
        print(f"❌ ERREUR: Aucune image dans {images_dir}")
        sys.exit(1)
    
    try:
        annotator = CandyBatchAnnotator(
            images_dir=images_dir,
            base_model='yolov8n.pt',
            project_root=project_root
        )
        annotator.annotate_all()
        
        print(f"\n{'='*60}")
        print(f"✅ PROCESSUS TERMINÉ")
        print(f"{'='*60}")
        print(f"📁 Annotations: {annotator.annotations_file}")
        print(f"📁 Dataset YOLO: {annotator.yolo_dataset_dir}")
        print(f"{'='*60}\n")
    
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
