#!/usr/bin/env python3
"""
Script simple pour vérifier les annotations sans interface graphique
Liste les images avec problèmes potentiels
"""

import os
from pathlib import Path

def check_annotations(images_dir, labels_dir):
    """Vérifie les annotations et trouve les problèmes"""
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Lister toutes les images
    image_files = sorted(list(images_dir.glob('*.jpg')) + 
                        list(images_dir.glob('*.png')))
    
    print("="*60)
    print("🔍 VÉRIFICATION DES ANNOTATIONS YOLO")
    print("="*60)
    print(f"📁 Dossier images: {images_dir}")
    print(f"📁 Dossier labels: {labels_dir}")
    print(f"📊 Total d'images: {len(image_files)}")
    print()
    
    # Statistiques
    no_labels = []
    few_labels = []
    many_labels = []
    total_annotations = 0
    
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + '.txt')
        
        if not label_path.exists():
            no_labels.append(img_path.name)
            continue
        
        # Compter les annotations
        with open(label_path, 'r') as f:
            annotations = [line.strip() for line in f if line.strip()]
            num_ann = len(annotations)
            total_annotations += num_ann
            
            if num_ann == 0:
                no_labels.append(img_path.name)
            elif num_ann <= 2:
                few_labels.append((img_path.name, num_ann))
            elif num_ann >= 15:
                many_labels.append((img_path.name, num_ann))
    
    # Résultats
    print(f"✅ Images avec labels: {len(image_files) - len(no_labels)}")
    print(f"❌ Images SANS labels: {len(no_labels)}")
    print(f"📊 Total annotations: {total_annotations}")
    if len(image_files) > 0:
        print(f"📊 Moyenne: {total_annotations / len(image_files):.1f} annotations/image")
    print()
    
    # Images sans labels
    if no_labels:
        print(f"❌ {len(no_labels)} images SANS annotations:")
        for i, img in enumerate(no_labels[:20], 1):  # Afficher max 20
            print(f"   {i}. {img}")
        if len(no_labels) > 20:
            print(f"   ... et {len(no_labels) - 20} autres")
        print()
    
    # Images avec peu de labels (possibles problèmes)
    if few_labels:
        print(f"⚠️  {len(few_labels)} images avec PEU d'annotations (≤2):")
        for i, (img, count) in enumerate(few_labels[:20], 1):
            print(f"   {i}. {img} ({count} boxes)")
        if len(few_labels) > 20:
            print(f"   ... et {len(few_labels) - 20} autres")
        print()
    
    # Images avec beaucoup de labels
    if many_labels:
        print(f"📦 {len(many_labels)} images avec BEAUCOUP d'annotations (≥15):")
        for i, (img, count) in enumerate(many_labels[:10], 1):
            print(f"   {i}. {img} ({count} boxes)")
        if len(many_labels) > 10:
            print(f"   ... et {len(many_labels) - 10} autres")
        print()
    
    # Distribution par classe
    print("📊 Distribution des classes:")
    class_counts = {i: 0 for i in range(6)}
    classes = ['Croco', 'Dragibus', 'Oeuf', 'Ourson', 'Schtroumpf', 'Tagada']
    
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if 0 <= class_id < 6:
                            class_counts[class_id] += 1
    
    for class_id, count in class_counts.items():
        if class_id < len(classes):
            bar = '█' * (count // 10) if count > 0 else ''
            print(f"   {class_id} - {classes[class_id]:12s}: {count:4d} {bar}")
    
    print()
    print("="*60)
    
    # Recommandations
    if no_labels or few_labels:
        print("\n💡 RECOMMANDATIONS:")
        if no_labels:
            print("   • Annotez les images sans labels")
        if few_labels:
            print("   • Vérifiez les images avec peu d'annotations")
        print("   • Utilisez check_annotations.bat pour éditer graphiquement")
        print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Vérifier les annotations YOLO')
    parser.add_argument('--dataset', default='datasets/yolo_dataset/train',
                       help='Chemin vers le dataset')
    
    args = parser.parse_args()
    
    images_dir = os.path.join(args.dataset, 'images')
    labels_dir = os.path.join(args.dataset, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"❌ Dossier images introuvable: {images_dir}")
        return
    
    if not os.path.exists(labels_dir):
        print(f"❌ Dossier labels introuvable: {labels_dir}")
        return
    
    check_annotations(images_dir, labels_dir)

if __name__ == "__main__":
    main()
