@echo off
REM Vérification rapide des annotations (sans interface graphique)

echo ========================================
echo   Verification des annotations - TRAIN
echo ========================================
echo.

wsl python3 scripts/verify_annotations.py --dataset datasets/yolo_dataset/train

echo.
echo ========================================
echo   Verification des annotations - VAL
echo ========================================
echo.

wsl python3 scripts/verify_annotations.py --dataset datasets/yolo_dataset/val

pause
