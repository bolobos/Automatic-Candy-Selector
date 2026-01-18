@echo off
REM Vérificateur d'annotations - Version Windows

echo ========================================
echo   Editeur d'annotations YOLO - Windows
echo ========================================
echo.

REM Vérifier si Python Windows est disponible
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Python n'est pas installe sur Windows ou n'est pas dans PATH
    echo.
    echo Solutions:
    echo 1. Installer Python depuis python.org
    echo 2. Ou utiliser le Microsoft Store
    pause
    exit /b 1
)

echo [INFO] Verification de Python et OpenCV...
python -c "import cv2" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INSTALLATION] Installation d'OpenCV et NumPy...
    pip install opencv-python numpy
    echo.
)

echo [DEMARRAGE] Lancement de l'editeur...
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

cd /d "%~dp0"
python scripts\check_annotations_windows.py

pause
