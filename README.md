# 🍬 Automatic Candy Selector

Système de détection automatique de bonbons utilisant YOLOv8.

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

### Application Java (Détection)

```bash
cd ML_Project_Candy/java_app
bash compile.sh                # Compilation (1 seule fois)
bash run.sh                    # Lancer l'application
```

L'application propose un menu pour choisir une image et affiche les bonbons détectés.

### Entraînement Python (si besoin)

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
