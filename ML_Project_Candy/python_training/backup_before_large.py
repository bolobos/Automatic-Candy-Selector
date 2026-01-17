import shutil
import os
from datetime import datetime
from pathlib import Path

def backup_all():
    """Sauvegarde complète avant entraînement YOLOv8l"""
    
    # Timestamp pour la sauvegarde
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/backup_yolov8n_{timestamp}"
    
    print(f"🔄 Création de la sauvegarde dans : {backup_dir}")
    os.makedirs(backup_dir, exist_ok=True)
    
    # 1. Sauvegarder tous les modèles .pt
    print("📁 Sauvegarde des modèles .pt...")
    if os.path.exists("yolo_models"):
        shutil.copytree("yolo_models", os.path.join(backup_dir, "yolo_models"))
        print("   ✅ yolo_models/ sauvegardé")
    
    # 2. Sauvegarder le modèle ONNX
    print("📁 Sauvegarde du modèle ONNX...")
    if os.path.exists("yolo_models/candy_yolov8.onnx"):
        shutil.copy("yolo_models/candy_yolov8.onnx", 
                   os.path.join(backup_dir, "candy_yolov8n.onnx"))
        print("   ✅ candy_yolov8.onnx sauvegardé")
    
    # 3. Sauvegarder tous les runs d'entraînement (dossier parent)
    print("📁 Sauvegarde des logs d'entraînement...")
    runs_path = "../../runs"
    if os.path.exists(runs_path):
        shutil.copytree(runs_path, os.path.join(backup_dir, "runs"))
        print("   ✅ runs/ sauvegardé")
    
    # 4. Sauvegarder les fichiers de configuration
    print("📁 Sauvegarde des configurations...")
    config_files = ["candy.yaml"]
    config_dir = os.path.join(backup_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    for file in config_files:
        if os.path.exists(file):
            shutil.copy(file, os.path.join(config_dir, file))
            print(f"   ✅ {file} sauvegardé")
    
    # Sauvegarder les scripts d'entraînement si présents
    for script in ["train_yolov8_candy.py", "export_adaptive_to_onnx.py", "train_adaptive_model.py"]:
        script_path = f"scripts/{script}"
        if os.path.exists(script_path):
            shutil.copy(script_path, os.path.join(config_dir, script))
            print(f"   ✅ scripts/{script} sauvegardé")
    
    # 5. Créer un fichier README avec les infos
    print("📝 Création du README...")
    readme_content = f"""# Backup YOLOv8n - {timestamp}

## Modèle sauvegardé
- **Architecture**: YOLOv8n (nano)
- **Date**: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
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
cp -r {backup_dir}/yolo_models ./

# Restaurer les logs
cp -r {backup_dir}/runs ../../

# Restaurer la config
cp {backup_dir}/config/candy.yaml ./
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
"""
    
    with open(os.path.join(backup_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("   ✅ README.md créé")
    
    # 6. Résumé de la sauvegarde
    print("\n" + "="*60)
    print("✅ SAUVEGARDE COMPLÉTÉE")
    print("="*60)
    print(f"📂 Emplacement : {os.path.abspath(backup_dir)}")
    
    # Calcul de la taille
    total_size = sum(f.stat().st_size for f in Path(backup_dir).rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    print(f"💾 Taille totale : {size_mb:.2f} MB")
    
    print("\n📋 Structure sauvegardée :")
    for root, dirs, files in os.walk(backup_dir):
        level = root.replace(backup_dir, '').count(os.sep)
        if level > 2:  # Limite la profondeur d'affichage
            continue
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # Afficher max 3 fichiers par dossier
                print(f'{subindent}📄 {file}')
            if len(files) > 3:
                print(f'{subindent}... et {len(files) - 3} autres fichiers')
    
    print("\n✅ Vous pouvez maintenant lancer l'entraînement YOLOv8l en toute sécurité !")
    print(f"📌 Backup path: {os.path.abspath(backup_dir)}")
    
    return backup_dir

if __name__ == "__main__":
    backup_all()
