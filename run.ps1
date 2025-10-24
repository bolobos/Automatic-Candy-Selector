# Script pour compiler et exécuter avec OpenCV
# Modifiez OPENCV_DIR si OpenCV est installé ailleurs

$OPENCV_DIR = "C:\opencv\opencv\build"
$OPENCV_JAR = "$OPENCV_DIR\java\opencv-4120.jar"
$OPENCV_LIB = "$OPENCV_DIR\java\x64"

# Vérifier si OpenCV existe
if (-not (Test-Path $OPENCV_DIR)) {
    Write-Host "ERREUR: OpenCV non trouvé dans $OPENCV_DIR" -ForegroundColor Red
    Write-Host "Veuillez télécharger et installer OpenCV depuis https://opencv.org/releases/" -ForegroundColor Yellow
    Write-Host "Extrayez-le dans C:\opencv" -ForegroundColor Yellow
    exit 1
}

# Trouver le fichier JAR OpenCV
$jarFiles = Get-ChildItem -Path "$OPENCV_DIR\java" -Filter "opencv-*.jar" -ErrorAction SilentlyContinue
if ($jarFiles.Count -eq 0) {
    Write-Host "ERREUR: Fichier JAR OpenCV non trouvé dans $OPENCV_DIR\java" -ForegroundColor Red
    exit 1
}
$OPENCV_JAR = $jarFiles[0].FullName
Write-Host "Utilisation de: $OPENCV_JAR" -ForegroundColor Green

# Compilation
Write-Host "`nCompilation..." -ForegroundColor Cyan
javac -cp ".;$OPENCV_JAR" ColorIdentificator3000.java

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation réussie!" -ForegroundColor Green
    
    # Exécution
    Write-Host "`nExécution..." -ForegroundColor Cyan
    java -cp ".;$OPENCV_JAR" "-Djava.library.path=$OPENCV_LIB" ColorIdentificator3000
} else {
    Write-Host "Erreur de compilation!" -ForegroundColor Red
}
