#!/usr/bin/env python3
"""
Script de vérification et correction des annotations YOLO
Permet de visualiser, modifier, supprimer et ajouter des bounding boxes
"""

import cv2
import os
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
import argparse

class AnnotationEditor:
    def __init__(self, images_dir, labels_dir, classes_file, start_index=0):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.classes = self.load_classes(classes_file)
        
        # Liste des images
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                  list(self.images_dir.glob('*.png')))
        self.current_index = max(0, min(start_index, len(self.image_files) - 1))
        
        # État de l'éditeur
        self.current_image = None
        self.current_image_resized = None
        self.scale_factor = 1.0
        self.current_annotations = []
        self.modified = False
        
        # Taille max de l'IMAGE seule (sans compter header ~85px + footer ~110px)
        self.max_width = 1200
        self.max_height = 600  # Fenêtre totale ~795px (600+85+110)
        
        # Interaction souris
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.resize_corner = None  # Coin/bord en cours de redimensionnement
        self.selected_box = None
        self.start_point = None
        self.current_class = 0
        self.resize_threshold = 10  # Distance pour détecter les coins/bords
        self.image_x_offset = 0  # Offset horizontal pour images portrait centrées
        
        # Couleurs pour chaque classe (dans l'ordre du fichier classes)
        self.colors = [
            (0, 255, 255),  # 0: Tagada - Jaune
            (0, 255, 0),    # 1: Dragibus - Vert
            (255, 255, 0),  # 2: Ourson - Cyan
            (0, 0, 255),    # 3: Oeuf - Bleu
            (255, 0, 0),    # 4: Croco - Rouge
            (255, 0, 255),  # 5: Schtroumpf - Magenta
        ]
        
        # Backup des labels modifiés
        self.backup_dir = Path("backups/annotations_backup_" + 
                               datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        print("🎨 Éditeur d'annotations YOLO")
        print(f"📁 Images: {self.images_dir}")
        print(f"📁 Labels: {self.labels_dir}")
        print(f"📊 Total d'images: {len(self.image_files)}")
        print(f"🎯 Classes: {', '.join(self.classes)}")
        
    def load_classes(self, classes_file):
        """Charge les noms des classes"""
        if not os.path.exists(classes_file):
            return ['Croco', 'Dragibus', 'Oeuf', 'Ourson', 'Schtroumpf', 'Tagada']
        
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def load_annotations(self, image_path):
        """Charge les annotations d'une image"""
        label_path = self.labels_dir / (image_path.stem + '.txt')
        annotations = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append([class_id, x_center, y_center, width, height])
        
        return annotations
    
    def save_annotations(self, image_path, annotations):
        """Sauvegarde les annotations"""
        label_path = self.labels_dir / (image_path.stem + '.txt')
        
        # Backup de l'ancien fichier
        if label_path.exists() and self.modified:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = self.backup_dir / label_path.name
            shutil.copy(label_path, backup_path)
        
        # Écriture des nouvelles annotations
        with open(label_path, 'w') as f:
            for ann in annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    def yolo_to_bbox(self, x_center, y_center, width, height, img_w, img_h):
        """Convertit YOLO format vers bbox (x1, y1, x2, y2) pour l'image redimensionnée"""
        # Utiliser les dimensions de l'image redimensionnée
        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)
        return x1, y1, x2, y2
    
    def bbox_to_yolo(self, x1, y1, x2, y2, img_w, img_h):
        """Convertit bbox vers YOLO format pour l'image redimensionnée"""
        # Utiliser les dimensions de l'image redimensionnée
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h
        return x_center, y_center, width, height
    
    def resize_image(self, image):
        """Redimensionne l'image pour qu'elle rentre à l'écran"""
        h, w = image.shape[:2]
        
        # Calculer le facteur de redimensionnement
        scale_w = self.max_width / w if w > self.max_width else 1.0
        scale_h = self.max_height / h if h > self.max_height else 1.0
        self.scale_factor = min(scale_w, scale_h)
        
        if self.scale_factor < 1.0:
            new_w = int(w * self.scale_factor)
            new_h = int(h * self.scale_factor)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def point_in_bbox(self, point, bbox):
        """Vérifie si un point est dans une bbox"""
        x, y = point
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_resize_corner(self, point, bbox):
        """Détermine si le point est sur un coin ou bord de la bbox
        Retourne: 'tl', 'tr', 'bl', 'br' pour les coins
                  't', 'b', 'l', 'r' pour les bords
                  None si pas sur un coin/bord
        """
        x, y = point
        x1, y1, x2, y2 = bbox
        threshold = self.resize_threshold
        
        # Coins (prioritaires)
        if abs(x - x1) <= threshold and abs(y - y1) <= threshold:
            return 'tl'  # Top-left
        elif abs(x - x2) <= threshold and abs(y - y1) <= threshold:
            return 'tr'  # Top-right
        elif abs(x - x1) <= threshold and abs(y - y2) <= threshold:
            return 'bl'  # Bottom-left
        elif abs(x - x2) <= threshold and abs(y - y2) <= threshold:
            return 'br'  # Bottom-right
        
        # Bords
        elif abs(y - y1) <= threshold and x1 <= x <= x2:
            return 't'  # Top
        elif abs(y - y2) <= threshold and x1 <= x <= x2:
            return 'b'  # Bottom
        elif abs(x - x1) <= threshold and y1 <= y <= y2:
            return 'l'  # Left
        elif abs(x - x2) <= threshold and y1 <= y <= y2:
            return 'r'  # Right
        
        return None
    
    def draw_annotations(self, image):
        """Dessine les annotations sur l'image"""
        img = image.copy()
        h, w = img.shape[:2]
        
        for i, ann in enumerate(self.current_annotations):
            class_id = ann[0]
            x1, y1, x2, y2 = self.yolo_to_bbox(ann[1], ann[2], ann[3], ann[4], w, h)
            
            # Couleur selon la sélection
            color = self.colors[class_id % len(self.colors)]
            thickness = 3 if i == self.selected_box else 2
            
            # Dessiner la bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            # Si sélectionnée, dessiner les poignées de redimensionnement
            if i == self.selected_box:
                handle_size = 6
                # Coins
                cv2.circle(img, (x1, y1), handle_size, color, -1)  # Top-left
                cv2.circle(img, (x2, y1), handle_size, color, -1)  # Top-right
                cv2.circle(img, (x1, y2), handle_size, color, -1)  # Bottom-left
                cv2.circle(img, (x2, y2), handle_size, color, -1)  # Bottom-right
                # Points du milieu des bords
                cv2.circle(img, ((x1+x2)//2, y1), handle_size//2, color, -1)  # Top
                cv2.circle(img, ((x1+x2)//2, y2), handle_size//2, color, -1)  # Bottom
                cv2.circle(img, (x1, (y1+y2)//2), handle_size//2, color, -1)  # Left
                cv2.circle(img, (x2, (y1+y2)//2), handle_size//2, color, -1)  # Right
            
            # Label
            label = f"{self.classes[class_id]}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """Gestion des événements souris"""
        if self.current_image_resized is None:
            return
        
        # Ajuster les coordonnées pour le décalage du header (85px) et offset horizontal
        header_height = 85
        y = y - header_height
        x = x - self.image_x_offset
        
        # Ignorer les clics dans le header ou footer ou en dehors de l'image
        h, w = self.current_image_resized.shape[:2]
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        
        # Bouton gauche pressé
        if event == cv2.EVENT_LBUTTONDOWN:
            # Vérifier si on clique sur une box existante ou ses coins/bords
            clicked_box = None
            resize_corner = None
            
            for i, ann in enumerate(self.current_annotations):
                x1, y1, x2, y2 = self.yolo_to_bbox(ann[1], ann[2], ann[3], ann[4], w, h)
                
                # D'abord vérifier les coins/bords pour le redimensionnement
                corner = self.get_resize_corner((x, y), (x1, y1, x2, y2))
                if corner:
                    clicked_box = i
                    resize_corner = corner
                    break
                # Sinon vérifier si on est dans la box pour le déplacement
                elif self.point_in_bbox((x, y), (x1, y1, x2, y2)):
                    clicked_box = i
                    break
            
            if clicked_box is not None:
                self.selected_box = clicked_box
                if resize_corner:
                    # Mode redimensionnement
                    self.resizing = True
                    self.resize_corner = resize_corner
                else:
                    # Mode déplacement
                    self.moving = True
                self.start_point = (x, y)
            else:
                # Commencer à dessiner une nouvelle box
                self.drawing = True
                self.start_point = (x, y)
                self.selected_box = None
        
        # Déplacement de la souris
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.resizing and self.selected_box is not None:
                # Redimensionner la box sélectionnée
                ann = self.current_annotations[self.selected_box]
                x1, y1, x2, y2 = self.yolo_to_bbox(ann[1], ann[2], ann[3], ann[4], w, h)
                
                # Calculer les nouvelles coordonnées selon le coin/bord
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
                
                if self.resize_corner in ['tl', 't', 'tr']:
                    new_y1 = max(0, min(y, y2 - 10))
                if self.resize_corner in ['bl', 'b', 'br']:
                    new_y2 = max(y1 + 10, min(h, y))
                if self.resize_corner in ['tl', 'l', 'bl']:
                    new_x1 = max(0, min(x, x2 - 10))
                if self.resize_corner in ['tr', 'r', 'br']:
                    new_x2 = max(x1 + 10, min(w, x))
                
                # S'assurer que les coordonnées sont valides
                if new_x1 < new_x2 and new_y1 < new_y2:
                    x_center, y_center, width, height = self.bbox_to_yolo(
                        new_x1, new_y1, new_x2, new_y2, w, h)
                    
                    self.current_annotations[self.selected_box][1] = x_center
                    self.current_annotations[self.selected_box][2] = y_center
                    self.current_annotations[self.selected_box][3] = width
                    self.current_annotations[self.selected_box][4] = height
                    self.modified = True
                
            elif self.moving and self.selected_box is not None:
                # Déplacer la box sélectionnée
                ann = self.current_annotations[self.selected_box]
                x1, y1, x2, y2 = self.yolo_to_bbox(ann[1], ann[2], ann[3], ann[4], w, h)
                
                dx = x - self.start_point[0]
                dy = y - self.start_point[1]
                
                new_x1 = max(0, min(w, x1 + dx))
                new_y1 = max(0, min(h, y1 + dy))
                new_x2 = max(0, min(w, x2 + dx))
                new_y2 = max(0, min(h, y2 + dy))
                
                x_center, y_center, width, height = self.bbox_to_yolo(
                    new_x1, new_y1, new_x2, new_y2, w, h)
                
                self.current_annotations[self.selected_box][1] = x_center
                self.current_annotations[self.selected_box][2] = y_center
                self.start_point = (x, y)
                self.modified = True
        
        # Bouton gauche relâché
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                # Terminer le dessin d'une nouvelle box
                if self.start_point:
                    x1, y1 = self.start_point
                    x2, y2 = x, y
                    
                    # S'assurer que x1 < x2 et y1 < y2
                    if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # Taille minimale
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        x_center, y_center, width, height = self.bbox_to_yolo(
                            x1, y1, x2, y2, w, h)
                        
                        self.current_annotations.append([
                            self.current_class, x_center, y_center, width, height
                        ])
                        self.modified = True
                
                self.drawing = False
                self.start_point = None
            
            if self.moving:
                self.moving = False
            
            if self.resizing:
                self.resizing = False
                self.resize_corner = None
    
    def show_help(self):
        """Affiche l'aide"""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║              ÉDITEUR D'ANNOTATIONS YOLO                      ║
╠══════════════════════════════════════════════════════════════╣
║ SOURIS:                                                      ║
║  • Clic gauche + Glisser     : Dessiner nouvelle box         ║
║  • Clic sur box + Glisser    : Déplacer box                  ║
║  • Clic coin/bord + Glisser  : Redimensionner box            ║
║                                                              ║
║ CLAVIER:                                                     ║
║  • ESPACE / Flèche droite    : Image suivante                ║
║  • Flèche gauche             : Image précédente              ║
║  • 0-5                       : Changer classe (ou classe     ║
║                                de la box sélectionnée)       ║
║  • d / Suppr                 : Supprimer box sélectionnée    ║
║  • s                         : Sauvegarder                   ║
║  • r                         : Recharger (annuler modifs)    ║
║  • h                         : Afficher cette aide           ║
║  • q / ESC                   : Quitter                       ║
╠══════════════════════════════════════════════════════════════╣
║ CLASSES:                                                     ║
║  0: Tagada      1: Dragibus     2: Ourson                    ║
║  3: Oeuf        4: Croco        5: Schtroumpf                ║
╠══════════════════════════════════════════════════════════════╣
║ REDIMENSIONNEMENT:                                           ║
║  • Les poignées (cercles) apparaissent sur la box active     ║
║  • Gros cercles = coins (redim. diagonale)                   ║
║  • Petits cercles = bords (redim. horizontal/vertical)       ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(help_text)
    
    def display_image(self):
        """Affiche l'image courante avec annotations"""
        if self.current_index >= len(self.image_files):
            print("✅ Toutes les images ont été vérifiées !")
            return False
        
        # Charger l'image
        image_path = self.image_files[self.current_index]
        self.current_image = cv2.imread(str(image_path))
        
        if self.current_image is None:
            print(f"❌ Erreur de chargement: {image_path}")
            return True
        
        # Redimensionner l'image pour l'affichage
        self.current_image_resized = self.resize_image(self.current_image)
        
        # Charger les annotations
        self.current_annotations = self.load_annotations(image_path)
        self.selected_box = None
        self.modified = False
        
        # Info
        h_orig, w_orig = self.current_image.shape[:2]
        h_disp, w_disp = self.current_image_resized.shape[:2]
        
        print(f"\n{'='*60}")
        print(f"📷 Image {self.current_index + 1}/{len(self.image_files)}: {image_path.name}")
        print(f"📐 Taille: {w_orig}x{h_orig} → {w_disp}x{w_disp} (échelle: {self.scale_factor:.2f})")
        print(f"📦 Annotations: {len(self.current_annotations)}")
        print(f"🎨 Classe courante: {self.current_class} - {self.classes[self.current_class]}")
        print(f"{'='*60}")
        
        return True
    
    def create_display(self, img_with_annotations):
        """Crée l'affichage complet avec header et footer"""
        h, w = img_with_annotations.shape[:2]
        
        # Largeur minimale pour afficher toutes les infos correctement
        min_width = 900
        display_width = max(w, min_width)
        
        # Créer une nouvelle image avec de l'espace pour header et footer
        header_height = 85
        footer_height = 80
        total_height = header_height + h + footer_height
        display = np.zeros((total_height, display_width, 3), dtype=np.uint8)
        
        # Centrer l'image si elle est plus étroite que la largeur minimale
        x_offset = (display_width - w) // 2
        self.image_x_offset = x_offset  # Sauvegarder pour ajuster les clics souris
        
        # Header noir
        cv2.rectangle(display, (0, 0), (display_width, header_height), (0, 0, 0), -1)
        
        # Ligne 1: Nom de l'image
        image_name = self.image_files[self.current_index].name
        cv2.putText(display, f"Fichier: {image_name}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # Ligne 2: Infos générales avec dimensions originales
        h_orig, w_orig = self.current_image.shape[:2]
        info_text = f"Image {self.current_index + 1}/{len(self.image_files)} | Taille: {w_orig}x{h_orig} | Boxes: {len(self.current_annotations)}"
        cv2.putText(display, info_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Ligne 3: Classe pour nouvelles boxes + classe de la box sélectionnée
        class_text = f"Nouvelle box: {self.classes[self.current_class]}"
        if self.selected_box is not None and self.selected_box < len(self.current_annotations):
            selected_class = self.current_annotations[self.selected_box][0]
            class_text += f" | Box selectionnee: {self.classes[selected_class]}"
        cv2.putText(display, class_text, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.modified:
            cv2.putText(display, "* MODIFIE *", (display_width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Copier l'image avec annotations au milieu (centrée si portrait)
        display[header_height:header_height+h, x_offset:x_offset+w] = img_with_annotations
        
        # Footer noir
        footer_y = header_height + h
        cv2.rectangle(display, (0, footer_y), (display_width, total_height), (0, 0, 0), -1)
        
        # Aide compacte sur 2 lignes
        y_offset = footer_y + 20
        cv2.putText(display, "Clic+Glisser=Box | Coins/Bords=Resize | ESPACE: Suivant | <-: Prec | 0-5: Classe | d: Suppr | s: Save | q: Quit", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(display, "Classes: 0:Tagada  1:Dragibus  2:Ourson  3:Oeuf  4:Croco  5:Schtroumpf | 0-5 sur box selectionnee change sa classe", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return display
    
    def run(self):
        """Lance l'éditeur"""
        self.show_help()
        
        cv2.namedWindow('Annotation Editor')
        cv2.setMouseCallback('Annotation Editor', self.mouse_callback)
        
        if not self.display_image():
            return
        
        while True:
            # Dessiner l'image avec annotations
            img_with_annotations = self.draw_annotations(self.current_image_resized)
            
            # Créer l'affichage complet
            display = self.create_display(img_with_annotations)
            
            h, w = self.current_image_resized.shape[:2]
            header_height = 85
            
            # Si en train de dessiner, dessiner le rectangle temporaire
            if self.drawing and self.start_point:
                try:
                    mouse_pos = cv2.getMousePos('Annotation Editor') if hasattr(cv2, 'getMousePos') else (-1, -1)
                    if mouse_pos[0] >= 0 and mouse_pos[1] >= 0:
                        # Ajuster pour le header et offset horizontal
                        mouse_x = mouse_pos[0] - self.image_x_offset
                        mouse_y = mouse_pos[1] - header_height
                        if 0 <= mouse_y < h and 0 <= mouse_x < w:
                            color = self.colors[self.current_class % len(self.colors)]
                            # Dessiner sur la partie image (après le header + offset)
                            start_x, start_y = self.start_point
                            cv2.rectangle(display, 
                                        (start_x + self.image_x_offset, start_y + header_height), 
                                        (mouse_x + self.image_x_offset, mouse_y + header_height), 
                                        color, 2)
                except:
                    pass
            
            cv2.imshow('Annotation Editor', display)
            
            # Gestion des touches
            key = cv2.waitKey(50) & 0xFF  # Augmenter le délai pour éviter les blocages
            
            # Quitter
            if key == ord('q') or key == 27:  # ESC
                if self.modified:
                    print("⚠️  Modifications non sauvegardées ! Appuyez sur 's' pour sauvegarder.")
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('s'):
                        image_path = self.image_files[self.current_index]
                        self.save_annotations(image_path, self.current_annotations)
                        print("✅ Annotations sauvegardées")
                break
            
            # Image suivante (Espace, flèche droite ou ->)
            elif key == ord(' ') or key == 83 or key == 3 or key == ord('n'):
                if self.modified:
                    image_path = self.image_files[self.current_index]
                    self.save_annotations(image_path, self.current_annotations)
                    print("✅ Annotations sauvegardées")
                
                self.current_index += 1
                if not self.display_image():
                    break
            
            # Image précédente (flèche gauche ou <-)
            elif key == 81 or key == 2 or key == ord('p'):
                if self.current_index > 0:
                    if self.modified:
                        image_path = self.image_files[self.current_index]
                        self.save_annotations(image_path, self.current_annotations)
                        print("✅ Annotations sauvegardées")
                    
                    self.current_index -= 1
                    self.display_image()
            
            # Changer de classe (0-5)
            elif ord('0') <= key <= ord('5'):
                new_class = key - ord('0')
                # Si une box est sélectionnée, changer sa classe
                if self.selected_box is not None and self.selected_box < len(self.current_annotations):
                    self.current_annotations[self.selected_box][0] = new_class
                    self.modified = True
                    print(f"✏️  Box sélectionnée changée en: {self.classes[new_class]}")
                else:
                    # Sinon, changer la classe pour les nouvelles boxes
                    self.current_class = new_class
                    print(f"🎨 Classe courante: {self.current_class} - {self.classes[self.current_class]}")
            
            # Supprimer box sélectionnée
            elif key == ord('d') or key == 127:  # d ou Delete
                if self.selected_box is not None and self.selected_box < len(self.current_annotations):
                    del self.current_annotations[self.selected_box]
                    self.selected_box = None
                    self.modified = True
                    print("🗑️  Box supprimée")
            
            # Sauvegarder
            elif key == ord('s'):
                if self.modified:
                    image_path = self.image_files[self.current_index]
                    self.save_annotations(image_path, self.current_annotations)
                    self.modified = False
                    print("✅ Annotations sauvegardées")
            
            # Recharger
            elif key == ord('r'):
                self.display_image()
                print("🔄 Annotations rechargées")
            
            # Aide
            elif key == ord('h'):
                self.show_help()
        
        cv2.destroyAllWindows()
        
        if self.backup_dir.exists():
            print(f"\n📦 Backup des modifications: {self.backup_dir}")
        print("👋 Éditeur fermé")


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Vérifier et corriger les annotations YOLO')
    parser.add_argument('--dataset', default='datasets/yolo_dataset/train',
                       help='Chemin vers le dataset (défaut: datasets/yolo_dataset/train)')
    parser.add_argument('--classes', default='datasets/label_names.txt',
                       help='Fichier des noms de classes')
    parser.add_argument('--start', type=int, default=0,
                       help='Numéro de l\'image de départ (défaut: 0)')
    
    args = parser.parse_args()
    
    images_dir = os.path.join(args.dataset, 'images')
    labels_dir = os.path.join(args.dataset, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"❌ Dossier images introuvable: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"❌ Dossier labels introuvable: {labels_dir}")
        return
    
    print(f"📁 {len(list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png')))} images trouvées")
    if args.start > 0:
        print(f"▶️  Démarrage à l'image numéro {args.start}")
    
    editor = AnnotationEditor(images_dir, labels_dir, args.classes, start_index=args.start)
    editor.run()


if __name__ == "__main__":
    main()
