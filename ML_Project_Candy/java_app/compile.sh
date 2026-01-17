#!/bin/bash
# Script de compilation Java pour WSL/Linux
# À exécuter depuis ML_Project_Candy/java_app/

set -e  # Arrêter en cas d'erreur

echo "☕ Compilation du détecteur YOLOv8..."

# Créer le dossier bin s'il n'existe pas
mkdir -p bin

# Compiler les classes
javac -cp "lib/opencv-4100.jar" \
      -d bin \
      src/YOLOv8CandyDetector.java \
      src/CandyDetectorApp.java

echo "✅ Compilation terminée !"
echo ""
echo "📝 Classes compilées :"
ls -lh bin/*.class 2>/dev/null | awk '{print "   -", $9, "(" $5 ")"}'
echo ""
echo "🚀 Pour exécuter :"
echo "   Sur Linux/WSL:"
echo "     export LD_LIBRARY_PATH=\$PWD/lib/opencv:\$LD_LIBRARY_PATH"
echo "     java -cp \"bin:lib/opencv-4100.jar\" -Djava.library.path=lib/opencv CandyDetectorApp"
echo ""
echo "   Sur Windows (PowerShell):"
echo "     java -cp \"bin;lib/opencv-4100.jar\" -Djava.library.path=lib/opencv CandyDetectorApp"
