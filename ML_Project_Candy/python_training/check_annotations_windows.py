# Script Python pour vérifier et corriger les annotations
# Version Windows native avec interface graphique

import os
import sys

# Ajouter le chemin du script
script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, script_dir)

# Importer et lancer
try:
    from check_and_fix_annotations import AnnotationEditor
    
    images_dir = "/mnt/c/Users/remic/Documents/Automatic-Candy-Selector/PXL_20260118_100006771_frames"
    labels_dir = "/mnt/c/Users/remic/Documents/Automatic-Candy-Selector/PXL_20260118_100006771_frames"
    classes_file = "datasets/label_names.txt"
    
    editor = AnnotationEditor(images_dir, labels_dir, classes_file)
    editor.run()
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("\nAssurez-vous que:")
    print("1. Python est installé sur Windows")
    print("2. opencv-python est installé: pip install opencv-python numpy")
    input("\nAppuyez sur Entrée pour quitter...")
