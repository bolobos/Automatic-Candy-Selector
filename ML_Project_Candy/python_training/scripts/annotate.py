#!/usr/bin/env python3
"""
Script d'annotation semi-automatique avec apprentissage adaptatif
Projet: Classification de bonbons - IN450/451
Auteur: Auto-généré
Version: 1.0 Production-ready
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
import signal

class CandyAnnotator:
    """Annotateur semi-automatique avec réentraînement adaptatif"""
    
    def __init__(self, images_dir, base_model='yolov8n.pt', project_root=None):
        """
        Initialisation de l'annotateur
        
        Args:
            images_dir: Dossier contenant les images à annoter
            base_model: Modèle YOLO de base
            project_root: Racine du projet (auto-détecté si None)
        """
        # Chemins
        if project_root is None:
            project_root = Path.home() / "candy_project"
        
        self.project_root = Path(project_root)
        self.images_dir = Path(images_dir)
        self.annotations_dir = self.project_root / "data" / "annotations"
        self.yolo_dataset_dir = self.project_root / "data" / "yolo_dataset"
        self.backups_dir = self.project_root / "data" / "backups"
        self.models_dir = self.project_root / "models"
        
        # Créer dossiers si nécessaire
        for d in [self.annotations_dir, self.backups_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Charger images
        self.image_files = self._load_image_files()
        self.total_images = len(self.image_files)
        
        if self.total_images == 0:
            raise ValueError(f"Aucune image trouvée dans {self.images_dir}")
        
        print(f"📸 {self.total_images} images détectées")
        
        # Charger ou créer annotations
        self.annotations_file = self.annotations_dir / "annotations.json"
        self.annotations = self._load_annotations()
        
        # Modèle YOLO
        pretrained_model = self.models_dir / "pretrained" / base_model
        if not pretrained_model.exists():
            print(f"⚠️ Modèle {pretrained_model} introuvable, utilisation modèle par défaut")
            pretrained_model = base_model
        
        self.model = YOLO(pretrained_model)
        self.current_model_path = pretrained_model
        
        # État annotation
        self.current_idx = self._get_first_unverified_index()
        self.verified_count = sum(1 for a in self.annotations if a.get('verified', False))
        self.retraining_thresholds = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900]
        
        # Interface OpenCV
        self.window_name = "Annotation Bonbons - Q: Quitter | ESPACE: Valider | R: Redessiner | ←→: Navigation"
        self.drawing = False
        self.bbox = None
        self.start_point = None
        
        # Gestion Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print(f"✅ Initialisé: {self.verified_count}/{self.total_images} déjà vérifiées")
    
    def _load_image_files(self):
        """Charge la liste des fichiers images"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        for ext in extensions:
            files.extend(self.images_dir.glob(f"*{ext}"))
            files.extend(self.images_dir.glob(f"*{ext.upper()}"))
        return sorted(files)
    
    def _load_annotations(self):
        """Charge les annotations existantes ou crée structure vide"""
        if self.annotations_file.exists():
            with open(self.annotations_file, 'r') as f:
                annotations = json.load(f)
            print(f"📂 Annotations chargées: {len(annotations)} entrées")
            return annotations
        else:
            # Créer structure vide
            annotations = []
            for img_file in self.image_files:
                annotations.append({
                    'image': img_file.name,
                    'bbox': None,
                    'verified': False,
                    'confidence': 0.0
                })
            return annotations
    
    def _get_first_unverified_index(self):
        """Trouve l'index de la première image non vérifiée"""
        for idx, ann in enumerate(self.annotations):
            if not ann.get('verified', False):
                return idx
        return 0  # Si tout vérifié, retourner au début
    
    def _signal_handler(self, sig, frame):
        """Gestion Ctrl+C propre"""
        print("\n\n⚠️ Interruption détectée - Sauvegarde en cours...")
        self.save_annotations()
        print("✅ Annotations sauvegardées. Au revoir!")
        sys.exit(0)
    
    def pre_annotate_batch(self, start_idx, batch_size=100):
        """
        Pré-annote un lot d'images avec le modèle actuel
        
        Args:
            start_idx: Index de départ
            batch_size: Nombre d'images à traiter
        """
        end_idx = min(start_idx + batch_size, self.total_images)
        print(f"\n🔍 Pré-annotation images {start_idx+1} à {end_idx}...")
        
        for idx in range(start_idx, end_idx):
            # Passer si déjà vérifié
            if self.annotations[idx].get('verified', False):
                continue
            
            img_file = self.image_files[idx]
            img = cv2.imread(str(img_file))
            
            if img is None:
                print(f"  ⚠️ Impossible de lire {img_file.name}")
                continue
            
            h, w = img.shape[:2]
            
            # Détection YOLO
            results = self.model(img, conf=0.25, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Prendre la box avec la plus grande surface
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                max_idx = np.argmax(areas)
                
                best_box = boxes[max_idx]
                conf = float(results[0].boxes.conf[max_idx])
                
                bbox = [int(best_box[0]), int(best_box[1]), 
                       int(best_box[2]), int(best_box[3])]
            else:
                # Box par défaut au centre
                bbox = [w//4, h//4, 3*w//4, 3*h//4]
                conf = 0.0
            
            # Mettre à jour annotation
            self.annotations[idx]['bbox'] = bbox
            self.annotations[idx]['confidence'] = conf
        
        print(f"✅ Pré-annotation terminée")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Callback souris pour dessiner bounding box"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.bbox = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.bbox[2] = x
                self.bbox[3] = y
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox[2] = x
            self.bbox[3] = y
            # Normaliser coordonnées (x1 < x2, y1 < y2)
            x1, y1, x2, y2 = self.bbox
            self.bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    
    def draw_interface(self, img, bbox, idx):
        """
        Dessine l'interface d'annotation
        
        Args:
            img: Image à afficher
            bbox: Bounding box [x1, y1, x2, y2]
            idx: Index image actuelle
        """
        display_img = img.copy()
        h, w = display_img.shape[:2]
        
        # Dessiner bounding box
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # HUD - Fond semi-transparent
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
        
        # Textes HUD
        verified = self.annotations[idx].get('verified', False)
        conf = self.annotations[idx].get('confidence', 0.0)
        
        # Ligne 1: Numéro image
        cv2.putText(display_img, f"Image {idx+1}/{self.total_images}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Ligne 2: Stats
        cv2.putText(display_img, f"Verifiees: {self.verified_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        cv2.putText(display_img, f"Confiance: {conf:.2f}", 
                   (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Coin supérieur droit: Statut
        status_text = "VERIFIE" if verified else "NON VERIFIE"
        status_color = (0, 255, 0) if verified else (0, 165, 255)
        status_symbol = "✓" if verified else "⚠"
        
        text = f"{status_symbol} {status_text}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(display_img, text, 
                   (w - text_size[0] - 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        return display_img
    
    def manual_correction(self):
        """
        Interface graphique pour correction manuelle
        Gère navigation, dessin, validation
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n🖱️ CONTRÔLES:")
        print("  Clic gauche + glisser : Dessiner bounding box")
        print("  ESPACE : Valider et passer à la suivante")
        print("  R : Redessiner (annuler box actuelle)")
        print("  ← → : Navigation entre images")
        print("  S : Sauvegarder progression")
        print("  Q : Quitter et sauvegarder")
        print("")
        
        while True:
            # Charger image actuelle
            img_file = self.image_files[self.current_idx]
            img = cv2.imread(str(img_file))
            
            if img is None:
                print(f"⚠️ Erreur lecture {img_file.name}, passage à la suivante")
                self.current_idx = (self.current_idx + 1) % self.total_images
                continue
            
            # Récupérer bbox courante
            current_bbox = self.annotations[self.current_idx].get('bbox')
            if current_bbox:
                self.bbox = current_bbox.copy()
            else:
                # Bbox par défaut
                h, w = img.shape[:2]
                self.bbox = [w//4, h//4, 3*w//4, 3*h//4]
            
            # Boucle affichage
            while True:
                display_img = self.draw_interface(img, self.bbox, self.current_idx)
                cv2.imshow(self.window_name, display_img)
                
                key = cv2.waitKey(1) & 0xFF
                
                # ESPACE: Valider
                if key == ord(' '):
                    self.annotations[self.current_idx]['bbox'] = self.bbox.copy()
                    if not self.annotations[self.current_idx].get('verified', False):
                        self.verified_count += 1
                    self.annotations[self.current_idx]['verified'] = True
                    
                    # Auto-save tous les 50
                    if self.verified_count % 50 == 0:
                        self.save_annotations()
                        print(f"💾 Auto-save: {self.verified_count} vérifiées")
                        
                        # Vérifier si réentraînement nécessaire
                        if self.verified_count in self.retraining_thresholds:
                            cv2.destroyWindow(self.window_name)
                            self.retrain_model()
                            # Pré-annoter prochain lot
                            next_batch_start = self.current_idx + 1
                            if next_batch_start < self.total_images:
                                self.pre_annotate_batch(next_batch_start, batch_size=100)
                            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(self.window_name, 1200, 800)
                            cv2.setMouseCallback(self.window_name, self.mouse_callback)
                    
                    # Passer à la suivante
                    self.current_idx = (self.current_idx + 1) % self.total_images
                    break
                
                # R: Redessiner
                elif key == ord('r') or key == ord('R'):
                    h, w = img.shape[:2]
                    self.bbox = [w//4, h//4, 3*w//4, 3*h//4]
                    self.drawing = False
                
                # Flèche gauche: Image précédente
                elif key == 81 or key == 2:  # Left arrow
                    self.current_idx = (self.current_idx - 1) % self.total_images
                    break
                
                # Flèche droite: Image suivante
                elif key == 83 or key == 3:  # Right arrow
                    self.current_idx = (self.current_idx + 1) % self.total_images
                    break
                
                # S: Save manuel
                elif key == ord('s') or key == ord('S'):
                    self.save_annotations()
                    print(f"💾 Sauvegarde manuelle: {self.verified_count}/{self.total_images}")
                
                # Q: Quitter
                elif key == ord('q') or key == ord('Q'):
                    cv2.destroyWindow(self.window_name)
                    self.save_annotations()
                    print(f"\n✅ Annotation terminée: {self.verified_count}/{self.total_images} vérifiées")
                    return
    
    def retrain_model(self):
        """
        Réentraîne YOLOv8 sur annotations vérifiées
        Améliore progressivement la pré-annotation
        """
        print(f"\n{'='*60}")
        print(f"🔄 RÉENTRAÎNEMENT ADAPTATIF - {self.verified_count} annotations")
        print(f"{'='*60}\n")
        
        # Préparer dataset YOLO
        verified_annotations = [a for a in self.annotations if a.get('verified', False)]
        
        if len(verified_annotations) < 10:
            print("⚠️ Pas assez d'annotations vérifiées (<10), réentraînement annulé")
            return
        
        # Créer structure YOLO
        train_dir = self.yolo_dataset_dir / "train"
        train_images_dir = train_dir / "images"
        train_labels_dir = train_dir / "labels"
        
        for d in [train_images_dir, train_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Copier images et créer labels
        print(f"📦 Préparation dataset: {len(verified_annotations)} images...")
        
        for ann in verified_annotations:
            img_name = ann['image']
            bbox = ann['bbox']
            
            # Copier image
            src_img = self.images_dir / img_name
            dst_img = train_images_dir / img_name
            shutil.copy2(src_img, dst_img)
            
            # Créer label YOLO
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
        yaml_content = f"""
path: {self.yolo_dataset_dir}
train: train/images
val: train/images

nc: 1
names: ['candy']
"""
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print("✅ Dataset préparé")
        
        # Entraîner modèle
        print(f"\n🚀 Entraînement en cours (10 epochs)...")
        print("   Cela peut prendre 2-5 minutes...\n")
        
        try:
            self.model.train(
                data=str(yaml_file),
                epochs=10,
                imgsz=640,
                batch=16,
                name='candy_adaptive',
                patience=3,
                save=True,
                exist_ok=True,
                verbose=False,
                plots=False
            )
            
            # Charger meilleur modèle
            best_model_path = Path("runs/detect/candy_adaptive/weights/best.pt")
            if best_model_path.exists():
                self.model = YOLO(str(best_model_path))
                self.current_model_path = best_model_path
                print(f"\n✅ Nouveau modèle chargé: {best_model_path}")
                print(f"   Pré-annotations suivantes seront plus précises!\n")
            else:
                print("⚠️ Modèle entraîné introuvable, conservation modèle actuel")
        
        except Exception as e:
            print(f"⚠️ Erreur entraînement: {e}")
            print("   Conservation modèle actuel")
        
        print(f"{'='*60}\n")
    
    def save_annotations(self):
        """Sauvegarde annotations en JSON avec backup"""
        # Backup précédent
        if self.annotations_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backups_dir / f"annotations_{timestamp}.json"
            shutil.copy2(self.annotations_file, backup_file)
        
        # Sauvegarder
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def run(self):
        """Pipeline complet d'annotation"""
        print(f"\n{'='*60}")
        print(f"🍬 DÉBUT ANNOTATION SEMI-AUTOMATIQUE ADAPTATIVE")
        print(f"{'='*60}\n")
        
        # Pré-annoter premier lot (ou reprendre où on était)
        if self.verified_count < 100:
            self.pre_annotate_batch(0, batch_size=100)
        elif self.current_idx < self.total_images:
            self.pre_annotate_batch(self.current_idx, batch_size=100)
        
        # Lancer interface correction
        self.manual_correction()
        
        # Sauvegarde finale
        self.save_annotations()
        
        print(f"\n{'='*60}")
        print(f"✅ ANNOTATION TERMINÉE")
        print(f"{'='*60}")
        print(f"📊 Résumé:")
        print(f"   Total images: {self.total_images}")
        print(f"   Vérifiées: {self.verified_count}")
        print(f"   Restantes: {self.total_images - self.verified_count}")
        print(f"\n📁 Annotations: {self.annotations_file}")
        print(f"{'='*60}\n")


def main():
    """Point d'entrée principal"""
    # Configuration
    project_root = Path.home() / "candy_project"
    images_dir = project_root / "data" / "raw_images"
    
    # Vérifications
    if not images_dir.exists() or not any(images_dir.iterdir()):
        print(f"❌ ERREUR: Aucune image dans {images_dir}")
        print(f"\n📝 Copiez vos images dans ce dossier:")
        print(f"   {images_dir}")
        print(f"\nExemple:")
        print(f"   cp /chemin/vers/vos/images/*.jpg {images_dir}/")
        sys.exit(1)
    
    # Lancer annotateur
    try:
        annotator = CandyAnnotator(
            images_dir=images_dir,
            base_model='yolov8n.pt',
            project_root=project_root
        )
        annotator.run()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interruption utilisateur")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
