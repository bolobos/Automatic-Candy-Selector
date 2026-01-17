@echo off
REM Script Windows pour lancer le détecteur YOLO en mode interactif
REM Double-cliquez sur ce fichier pour lancer le programme

cd /d "%~dp0"
wsl bash -c "cd /mnt/c/Users/remic/Documents/Automatic-Candy-Selector/ML_Project_Candy/java_app && ./run_interactive.sh"
pause
