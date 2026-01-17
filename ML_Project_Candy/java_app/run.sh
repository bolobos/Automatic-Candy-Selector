#!/bin/bash
# Script de lancement simple de CandyDetectorApp

# Configurer le chemin des bibliothèques OpenCV natives
export LD_LIBRARY_PATH="$PWD/lib/opencv:$LD_LIBRARY_PATH"

# Exécuter l'application
java -cp "bin:lib/*" CandyDetectorApp "$@"
