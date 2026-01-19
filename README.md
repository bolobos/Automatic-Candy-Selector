# 🍬 Automatic Candy Selector

Système intelligent de détection automatique de bonbons utilisant YOLOv8 et OpenCV.

## 📝 Description du projet

Ce projet détecte automatiquement 6 types de bonbons dans des images ou des vidéos en utilisant un modèle de deep learning YOLOv8. Une application Java avec OpenCV permet l'intégration dans un système de tri automatisé.

**Pipeline complet** :
1. **Extraction de frames** depuis des vidéos de bonbons
2. **Annotation automatique** avec un modèle pré-entraîné + correction manuelle via interface graphique
3. **Entraînement** d'un modèle YOLOv8m personnalisé (98% mAP50)
4. **Export ONNX** pour déploiement
5. **Détection en temps réel** en Java avec OpenCV

## 📁 Structure du projet

```
ML_Project_Candy/
├── java_app/                         # Application Java de détection
│   ├── src/
│   │   ├── CandyDetectorApp.java            # ⭐ Application principale
│   │   ├── YOLOv8CandyDetector.java         # Détection YOLO
│   │   ├── VideoFrameExtractor.java         # Extraction frames vidéo
│   │   ├── ColorIdentificator3000.java      # (Expérimental)
│   │   └── ImageClassifier.java             # (Expérimental)
│   ├── lib/opencv/                          # Bibliothèques OpenCV 4.10.0
│   ├── old_models/
│   │   ├── candy_yolov8.onnx               # Ancien modèle YOLOv8s
│   │   └── candy.names                      # Liste des classes
│   ├── compile.sh                           # Compilation Linux/WSL
│   ├── run.sh                              # ⭐ Exécution Linux/WSL
│   ├── DetecteurBonbons.bat                # Lancement Windows
│   └── GUIDE_UTILISATION.txt               # Instructions détaillées
│
├── models/
│   └── candy_yolov8m.onnx                   # ⭐ Modèle YOLOv8m entraîné (98% mAP50)
│
└── python_training/                   # Scripts d'entraînement Python
    ├── datasets/
    │   ├── label_names.txt                  # ⭐ 6 classes de bonbons
    │   └── yolo_dataset/                    # Dataset formaté YOLO
    │       ├── train/                       # 1294 images + labels
    │       ├── val/                         # 228 images + labels
    │       └── test/                        # Images test
    ├── scripts/                             # 📂 13 scripts Python
    │   ├── train_yolov8_candy.py           # ⭐ Entraînement principal
    │   ├── export_and_stats.py             # Export ONNX + métriques
    │   ├── check_and_fix_annotations.py    # Éditeur annotations
    │   ├── auto_label_images.py            # Auto-annotation batch
    │   └── ... (9 autres scripts)
    ├── candy.yaml                           # ⭐ Configuration YOLOv8
    ├── check_annotations.bat                # Vérification train (WSL+X11)
    ├── check_annotations_val.bat            # Vérification val (WSL+X11)
    └── check_annotations_WIN.bat            # Vérification Windows natif

runs/detect/
├── candy_detector_yolov8m7/             # Résultats modèle m7 (référence)
└── candy_detector_yolov8m14/            # ⭐ Résultats modèle actuel
```

## 🚀 Utilisation rapide

### 1. Détection sur une image (Java)

```bash
cd ML_Project_Candy/java_app

# Sous WSL/Linux
bash compile.sh                    # Compilation (1 seule fois)
bash run.sh                        # Lancer l'application

# Sous Windows
DetecteurBonbons.bat              # Lancement direct
```

L'application propose un menu pour choisir une image et affiche les bonbons détectés avec leurs bounding boxes. Le résultat est sauvegardé dans `detection_result.jpg`.

### 2. Extraction de frames depuis une vidéo

```bash
cd ML_Project_Candy/java_app
# Placer votre vidéo dans le dossier parent
# Modifier src/VideoFrameExtractor.java pour pointer vers le dossier de la vidéo
javac -cp 'lib/opencv-4100.jar' -d bin src/VideoFrameExtractor.java
export LD_LIBRARY_PATH=$PWD/lib/opencv:$LD_LIBRARY_PATH
java -cp 'bin:lib/opencv-4100.jar' -Djava.library.path=lib/opencv VideoFrameExtractor
```

Extrait 5 images par seconde dans un dossier `{nom_video}_frames/`.

### 3. Annotation manuelle des images

```bash
cd ML_Project_Candy/python_training

# Sous WSL (avec X11 server: VcXsrv ou Xming)
./check_annotations.bat              # Dataset train
./check_annotations_val.bat          # Dataset validation

# Sous Windows (Python natif)
./check_annotations_WIN.bat
```

Interface graphique pour vérifier et corriger les annotations :
- **Clic gauche + Glisser** : Créer/déplacer une bounding box
- **0-5** : Changer la classe (0:Tagada, 1:Dragibus, 2:Ourson, 3:Oeuf, 4:Croco, 5:Schtroumpf)
- **d** : Supprimer une box
- **ESPACE / n** : Image suivante
- **Flèche gauche / p** : Image précédente
- **s** : Sauvegarder
- **q** : Quitter

### 4. Entraînement du modèle (si besoin)

```bash
cd ML_Project_Candy/python_training
python scripts/train_yolov8_candy.py --model m --epochs 100

# Options disponibles:
# --model n/s/m/l/x  : Taille du modèle (nano/small/medium/large/xlarge)
# --epochs N         : Nombre d'epochs (défaut: 100)
# --imgsz N         : Taille des images (défaut: 640)
```

### 5. Export ONNX pour Java

```bash
python scripts/export_and_stats.py
# Génère: models/candy_yolov8m.onnx + statistiques complètes
```

## 🎯 Classes détectées

1. Tagada
2. Dragibus
3. Ourson
4. Oeuf
5. Croco
6. Schtroumpf

## 📊 Performances du modèle actuel

- **Modèle** : YOLOv8m (Medium) - `candy_detector_yolov8m14`
- **Entraînement** : 100 epochs (58 min sur RTX 3060 Ti)
- **mAP50** : 99.5% ⭐
- **mAP50-95** : 97.56%
- **Précision** : 98.63%
- **Recall** : 99.95%
- **Batch size** : 16
- **Image size** : 640x640
- **Fichier ONNX** : `ML_Project_Candy/models/candy_yolov8m.onnx`
- **Dataset** : 1522 images (1294 train / 228 val)
- **Classes** : 6 bonbons (Tagada, Dragibus, Ourson, Oeuf, Croco, Schtroumpf)

## ⚙️ Configuration système

- **OS** : Windows 11 avec WSL Ubuntu
- **Python** : 3.12+ avec ultralytics, opencv-python, numpy
- **Java** : OpenJDK avec OpenCV 4.10.0
- **GPU** : NVIDIA RTX 3060 Ti (recommandé pour entraînement)

## 📖 Documentation

- **README.md** : Vue d'ensemble du projet
- **ETAT_FINAL.txt** : État détaillé du projet, structure complète
- **Java** : [java_app/GUIDE_UTILISATION.txt](ML_Project_Candy/java_app/GUIDE_UTILISATION.txt)
- **Python** : Scripts documentés dans `python_training/scripts/`
- **Classes** : [python_training/datasets/label_names.txt](ML_Project_Candy/python_training/datasets/label_names.txt)
- **Config YOLO** : [python_training/candy.yaml](ML_Project_Candy/python_training/candy.yaml)

## 🎓 Crédits

**Projet IN450/451** - Machine Learning  
**École** : ESISAR - P2027APP
**Auteur** : Rémi C.

---

**Dernière mise à jour** : 18 janvier 2026

