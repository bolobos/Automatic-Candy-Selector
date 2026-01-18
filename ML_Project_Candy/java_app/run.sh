#!/bin/bash
# Script de lancement simple de CandyDetectorApp

# Récupérer le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configurer le chemin des bibliothèques OpenCV natives
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib/opencv:$LD_LIBRARY_PATH"

# Exécuter l'application
java -cp "$SCRIPT_DIR/bin:$SCRIPT_DIR/lib/opencv-4100.jar" -Djava.library.path="$SCRIPT_DIR/lib/opencv" CandyDetectorApp "$@"
