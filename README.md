# 🍬 Automatic Candy Selector

Système intelligent de détection et tri automatique de bonbons utilisant YOLOv8 et OpenCV.

## 📝 Description du projet

Ce projet permet de détecter automatiquement différents types de bonbons dans des images ou des vidéos en utilisant un modèle de deep learning YOLOv8. Le système extrait des frames de vidéos, détecte les bonbons présents, et peut potentiellement identifier leurs couleurs pour un tri automatisé.

**Pipeline complet** :
1. **Extraction de frames** depuis des vidéos de bonbons
2. **Annotation automatique** avec un modèle pré-entraîné
3. **Correction manuelle** des annotations via interface graphique
4. **Entraînement** d'un modèle YOLOv8 personnalisé
5. **Détection** en Java avec OpenCV pour intégration système
6. **Classification de couleurs** (théorique, en développement)

## 📁 Structure du projet

```
ML_Project_Candy/
├── java_app/                    # Application Java de détection
│   ├── src/
│   │   ├── CandyDetectorApp.java       # Application principale
│   │   └── YOLOv8CandyDetector.java    # Classe de détection
│   ├── models/
│   │   ├── candy_yolov8.onnx           # Modèle YOLOv8s entraîné (43 MB)
│   │   └── candy.names                 # Liste des classes
│   ├── compile.sh                      # Compilation
│   ├── run.sh                         # Exécution
│   └── GUIDE_UTILISATION.txt          # Instructions détaillées
│
└── python_training/             # Scripts d'entraînement Python
    ├── datasets/
    │   └── nos_dataset/
    │       ├── Entrainement/          # 1294 images (85%)
    │       └── Test/                   # 228 images (15%)
    ├── scripts/
    │   ├── train_yolov8_candy.py      # Script d'entraînement principal
    │   ├── export_and_stats.py        # Export ONNX + statistiques
    │   └── create_val_from_test.py    # Création validation set
    ├── yolo_models/
    │   └── candy_yolov8.onnx          # Modèle exporté
    └── candy.yaml                     # Configuration dataset YOLO
```

## 🚀 Utilisation rapide

### 1. Détection sur une image (Java)

```bash
cd ML_Project_Candy/java_app
wsl bash compile.sh                # Compilation (1 seule fois)
wsl bash run.sh                    # Lancer l'application
```

L'application propose un menu pour choisir une image et affiche les bonbons détectés avec leurs bounding boxes.

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
python check_annotations_windows.py
```

Interface graphique pour vérifier et corriger les annotations :
- Clic + Glisser : Créer une bounding box
- 0-5 : Changer la classe (0:Tagada, 1:Dragibus, 2:Ourson, 3:Oeuf, 4:Croco, 5:Schtroumpf)
- Sélectionner une box + 0-5 : Changer la classe de la box sélectionnée
- d : Supprimer une box
- ESPACE : Image suivante

### 4. Entraînement du modèle (si besoin)

```bash
cd ML_Project_Candy/python_training
source ~/yolo_training_env/bin/activate
python scripts/train_yolov8_candy.py --train --model s --epochs 100
```

## 🎯 Classes détectées

1. Tagada
2. Dragibus
3. Ourson
4. Oeuf
5. Croco
6. Schtroumpf

## 📊 Performances du modèle actuel

- **Modèle** : YOLOv8s (Small)
- **mAP50** : 97.85%
- **mAP50-95** : 94.96%
- **Précision** : 95.85%
- **Recall** : 96.05%
- **Temps d'entraînement** : 33 minutes (100 epochs)
- **Dataset** : 1522 images (1294 train / 228 val)

## ⚙️ Configuration système

- **OS** : Windows avec WSL Ubuntu
- **Python** : 3.12 (environnement virtuel : `~/yolo_training_env`)
- **Java** : OpenJDK avec OpenCV 4.10.0
- **GPU** : NVIDIA RTX 3060 Ti (optionnel)

## 📖 Documentation

- **Java** : Voir [java_app/GUIDE_UTILISATION.txt](ML_Project_Candy/java_app/GUIDE_UTILISATION.txt)
- **Python** : Scripts documentés dans `python_training/scripts/`

---

**Dernière mise à jour** : 16 janvier 2026
