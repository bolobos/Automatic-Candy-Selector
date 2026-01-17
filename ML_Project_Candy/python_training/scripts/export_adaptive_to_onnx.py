#!/usr/bin/env python3
"""
Export du modèle adaptatif YOLOv8 vers ONNX pour Java
Projet: Classification de bonbons - IN450/451
Version: 1.0 Production-ready
"""

import sys
from pathlib import Path
from ultralytics import YOLO

def export_to_onnx(model_path=None, output_dir=None):
    """
    Exporte le modèle YOLOv8 en format ONNX
    
    Args:
        model_path: Chemin vers le modèle .pt (auto-détecté si None)
        output_dir: Dossier de sortie (même que model_path si None)
    """
    print("="*60)
    print("📦 EXPORT MODÈLE YOLO → ONNX POUR JAVA")
    print("="*60)
    print()
    
    # Détecter meilleur modèle adaptatif
    if model_path is None:
        candidates = [
            Path("runs/detect/candy_adaptive/weights/best.pt"),
            Path.home() / "candy_project" / "models" / "adaptive" / "best.pt",
            Path("yolo_training/candy_detector/weights/best.pt")
        ]
        
        for candidate in candidates:
            if candidate.exists():
                model_path = candidate
                print(f"✓ Modèle détecté: {model_path}")
                break
        
        if model_path is None:
            print("❌ ERREUR: Aucun modèle trouvé")
            print("\nEmplacements cherchés:")
            for c in candidates:
                print(f"  - {c}")
            print("\nAssurez-vous d'avoir entraîné un modèle avant l'export.")
            sys.exit(1)
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"❌ ERREUR: Modèle introuvable: {model_path}")
            sys.exit(1)
    
    # Définir dossier sortie
    if output_dir is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Dossier sortie: {output_dir}")
    print()
    
    # Charger modèle
    print("🔄 Chargement modèle PyTorch...")
    try:
        model = YOLO(str(model_path))
        print("✓ Modèle chargé avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement: {e}")
        sys.exit(1)
    
    # Export ONNX
    print("\n🚀 Export ONNX en cours...")
    print("   (Cette opération peut prendre 1-2 minutes)")
    print()
    
    try:
        export_path = model.export(
            format='onnx',
            opset=12,          # Compatible ONNX Runtime Java
            simplify=True,     # Optimisations graphe
            dynamic=False,     # Input size fixe (plus rapide)
            imgsz=640,         # Taille standard YOLO
            half=False,        # Float32 (meilleure compatibilité)
            int8=False         # Pas de quantization
        )
        
        onnx_file = Path(export_path)
        
        # Infos fichier
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print(f"✅ EXPORT RÉUSSI")
        print(f"{'='*60}")
        print(f"📄 Fichier ONNX: {onnx_file}")
        print(f"💾 Taille: {size_mb:.2f} MB")
        print(f"\n🔧 Paramètres export:")
        print(f"   - Format: ONNX (opset 12)")
        print(f"   - Input size: 640x640")
        print(f"   - Précision: Float32")
        print(f"   - Optimisé: Oui (simplify=True)")
        print(f"\n📦 UTILISATION EN JAVA:")
        print(f"   1. Copier {onnx_file.name} dans votre projet Java")
        print(f"   2. Ajouter dépendance ONNX Runtime:")
        print(f"      <dependency>")
        print(f"          <groupId>com.microsoft.onnxruntime</groupId>")
        print(f"          <artifactId>onnxruntime</artifactId>")
        print(f"          <version>1.16.0</version>")
        print(f"      </dependency>")
        print(f"\n   3. Charger modèle:")
        print(f"      OrtEnvironment env = OrtEnvironment.getEnvironment();")
        print(f"      OrtSession session = env.createSession(\"{onnx_file.name}\");")
        print(f"\n   4. Voir GUIDE_JAVA.md pour code complet")
        print(f"{'='*60}\n")
        
        # Créer fichier .names pour classes
        names_file = output_dir / "candy.names"
        with open(names_file, 'w') as f:
            f.write("candy\n")
        print(f"✓ Fichier classes créé: {names_file}")
        
        return str(onnx_file)
    
    except Exception as e:
        print(f"\n❌ ERREUR EXPORT: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Point d'entrée principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export modèle YOLOv8 candy detector vers ONNX"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Chemin vers modèle .pt (auto-détecté si omis)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Dossier de sortie (même que modèle si omis)"
    )
    
    args = parser.parse_args()
    
    export_to_onnx(model_path=args.model, output_dir=args.output)


if __name__ == "__main__":
    main()
