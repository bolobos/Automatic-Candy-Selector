@echo off
REM Editeur d'annotations YOLO - Version WSL avec X11

echo ========================================
echo   Editeur d'annotations YOLO
echo ========================================
echo.

echo [INFO] Configuration de l'affichage X11...
echo.

REM Configurer DISPLAY pour WSL
wsl bash -c "export DISPLAY=:0"

echo [INFO] Lancement de l'editeur (Interface graphique)...
echo.
echo CONTROLES:
echo   - Clic gauche + Glisser : Dessiner nouvelle box
echo   - Clic sur box + Glisser : Deplacer box
echo   - ESPACE : Image suivante
echo   - Fleche gauche : Image precedente
echo   - 0-5 : Changer la classe courante
echo   - d : Supprimer box selectionnee
echo   - s : Sauvegarder
echo   - h : Aide complete
echo   - q : Quitter
echo.

echo [IMPORTANT] Si aucune fenetre ne s'ouvre:
echo 1. Installez VcXsrv ou Xming (serveur X pour Windows)
echo 2. Lancez XLaunch avec "Disable access control"
echo 3. Relancez ce script
echo.

REM Vérifier si un argument est fourni pour le numéro de départ
if "%1"=="" (
    echo Demarrage a l'image 0...
    echo.
    pause
    wsl bash -c "cd /mnt/c/Users/remic/Documents/Automatic-Candy-Selector/ML_Project_Candy/python_training && DISPLAY=:0 python3 scripts/check_and_fix_annotations.py --dataset datasets/yolo_dataset/train --classes datasets/label_names.txt"
) else (
    echo Demarrage a l'image numero %1...
    echo.
    pause
    wsl bash -c "cd /mnt/c/Users/remic/Documents/Automatic-Candy-Selector/ML_Project_Candy/python_training && DISPLAY=:0 python3 scripts/check_and_fix_annotations.py --dataset datasets/yolo_dataset/train --classes datasets/label_names.txt --start %1"
)

pause
