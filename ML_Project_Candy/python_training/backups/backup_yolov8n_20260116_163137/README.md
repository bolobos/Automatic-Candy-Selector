# Backup YOLOv8n - 20260116_163137

## Modèle sauvegardé
- **Architecture**: YOLOv8n (nano)
- **Date**: 16/01/2026 16:31:46
- **Raison**: Avant entraînement YOLOv8l

## Contenu de la sauvegarde
1. `yolo_models/` - Tous les modèles .pt entraînés
2. `candy_yolov8n.onnx` - Modèle exporté ONNX
3. `runs/` - Logs et métriques d'entraînement complets
4. `config/` - Fichiers de configuration (YAML, scripts)

## Structure des runs sauvegardés
- `runs/detect/candy_detector/` - Premier entraînement
- `runs/detect/candy_detector2/` - Deuxième entraînement
- `runs/detect/candy_detector3/` - Troisième entraînement
- `runs/detect/candy_detector4/` - Quatrième entraînement
- `runs/detect/candy_detector5/` - Cinquième entraînement
- `runs/detect/candy_detector6/` - Sixième entraînement
- `runs/detect/candy_detector7/` - Septième entraînement (final)
- `runs/detect/candy_adaptive/` - Entraînement adaptatif

## Restauration
Pour restaurer cette version :
```bash
# Restaurer les modèles
cp -r backups/backup_yolov8n_20260116_163137/yolo_models ./

# Restaurer les logs
cp -r backups/backup_yolov8n_20260116_163137/runs ../../

# Restaurer la config
cp backups/backup_yolov8n_20260116_163137/config/candy.yaml ./
```

## Métriques du modèle YOLOv8n (candy_detector7)
- Architecture: YOLOv8n
- Epochs: 100
- Batch size: 8-16
- Dataset: 6 classes de bonbons
- Augmentation: hsv_h=0.03, fliplr=0.5, mosaic=1.0

## Prochaine étape
Entraînement YOLOv8l avec :
- GPU: RTX 3060 Ti
- Batch size: 12-16
- Epochs: 100
- Même configuration d'augmentation
