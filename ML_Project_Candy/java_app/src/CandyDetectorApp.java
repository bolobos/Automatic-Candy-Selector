import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Application principale de détection de bonbons
 * 
 * Compilation:
 *   javac -cp "lib/opencv-4100.jar" -d bin src/CandyDetectorApp.java src/YOLOv8CandyDetector.java
 * 
 * Exécution (Windows):
 *   java -cp "bin;lib/opencv-4100.jar" -Djava.library.path=lib/opencv CandyDetectorApp
 * 
 * Exécution (Linux/WSL):
 *   export LD_LIBRARY_PATH=$PWD/lib/opencv:$LD_LIBRARY_PATH
 *   java -cp "bin:lib/opencv-4100.jar" -Djava.library.path=lib/opencv CandyDetectorApp
 */
public class CandyDetectorApp {
    
    /**
     * Liste les images disponibles dans un dossier
     */
    private static List<String> listImages(String folderPath) {
        File folder = new File(folderPath);
        if (!folder.exists() || !folder.isDirectory()) {
            return new ArrayList<>();
        }
        
        File[] files = folder.listFiles((dir, name) -> {
            String lower = name.toLowerCase();
            return lower.endsWith(".jpg") || lower.endsWith(".jpeg") || 
                   lower.endsWith(".png") || lower.endsWith(".bmp");
        });
        
        if (files == null) return new ArrayList<>();
        
        return Arrays.stream(files)
            .sorted(Comparator.comparing(File::getName))
            .map(File::getAbsolutePath)
            .collect(Collectors.toList());
    }
    
    public static void main(String[] args) {
        // Charger la bibliothèque OpenCV
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("\n❌ ERREUR: Impossible de charger OpenCV");
            System.err.println("Sur Windows: java -cp \"bin;lib/opencv-4100.jar\" -Djava.library.path=lib/opencv CandyDetectorApp");
            System.err.println("Sur Linux/WSL: export LD_LIBRARY_PATH=$PWD/lib/opencv:$LD_LIBRARY_PATH");
            System.err.println("              java -cp \"bin:lib/opencv-4100.jar\" -Djava.library.path=lib/opencv CandyDetectorApp");
            System.err.println("\nDétails de l'erreur:");
            e.printStackTrace();
            System.exit(1);
        }
        
        Scanner scanner = new Scanner(System.in);
        
        try {
            // Créer le détecteur
            YOLOv8CandyDetector detector = new YOLOv8CandyDetector(
                "models/candy_yolov8.onnx",
                "../python_training/datasets/label_names.txt"
            );
            
            System.out.println("\n" + "=".repeat(60));
            System.out.println("🍬 DÉTECTEUR DE BONBONS YOLOV8");
            System.out.println("=".repeat(60));
            
            String imagePath = null;
            
            // Si argument fourni, l'utiliser directement
            if (args.length > 0) {
                imagePath = args[0];
            } else {
                // Sinon, proposer un menu
                String testFolder = "../python_training/datasets/yolo_dataset/val/images";
                List<String> images = listImages(testFolder);
                
                System.out.println("\nOptions:");
                System.out.println("1. Choisir une image du dossier Test (" + images.size() + " images)");
                System.out.println("2. Saisir un chemin personnalisé");
                System.out.print("\nVotre choix (1 ou 2): ");
                
                String choice = scanner.nextLine().trim();
                
                if (choice.equals("1") && !images.isEmpty()) {
                    // Afficher les 15 premières images
                    System.out.println("\n📂 Images disponibles dans Test:");
                    int limit = Math.min(15, images.size());
                    for (int i = 0; i < limit; i++) {
                        String name = new File(images.get(i)).getName();
                        System.out.println("  " + (i + 1) + ". " + name);
                    }
                    if (images.size() > 15) {
                        System.out.println("  ... (" + (images.size() - 15) + " autres images)");
                    }
                    
                    System.out.print("\nNuméro de l'image (1-" + limit + ") ou nom de fichier: ");
                    String input = scanner.nextLine().trim();
                    
                    try {
                        int index = Integer.parseInt(input) - 1;
                        if (index >= 0 && index < images.size()) {
                            imagePath = images.get(index);
                        } else {
                            System.err.println("❌ Numéro invalide");
                            scanner.close();
                            return;
                        }
                    } catch (NumberFormatException e) {
                        // C'est un nom de fichier
                        imagePath = testFolder + "/" + input;
                    }
                    
                } else {
                    System.out.print("\n📸 Chemin complet de l'image: ");
                    imagePath = scanner.nextLine().trim();
                }
            }
            
            // Vérifier que le fichier existe
            if (imagePath == null || !new File(imagePath).exists()) {
                System.err.println("\n❌ Erreur: Image introuvable: " + imagePath);
                scanner.close();
                return;
            }
            
            // Détecter
            System.out.println("\n🔍 Analyse de: " + new File(imagePath).getName());
            System.out.println("⏳ Détection en cours...\n");
            
            // Charger l'image pour l'analyse des couleurs
            Mat image = Imgcodecs.imread(imagePath);
            
            List<YOLOv8CandyDetector.Detection> detections = detector.detect(image);
            
            // Analyser les couleurs de chaque détection
            if (!detections.isEmpty()) {
                System.out.println("🎨 Analyse des couleurs en cours...\n");
                detector.analyzeColorsForDetections(image, detections);
            }
            
            // Afficher les résultats
            System.out.println("=".repeat(60));
            System.out.println("🍬 RÉSULTATS DE DÉTECTION");
            System.out.println("=".repeat(60));
            
            if (detections.isEmpty()) {
                System.out.println("❌ Aucun bonbon détecté");
            } else {
                System.out.println("✅ " + detections.size() + " bonbon(s) détecté(s):\n");
                for (int i = 0; i < detections.size(); i++) {
                    YOLOv8CandyDetector.Detection det = detections.get(i);
                    System.out.printf("  %d. %s - %.1f%% de confiance\n", 
                        i + 1, det.className, det.confidence * 100);
                    System.out.printf("     Position: [%.0f, %.0f] Taille: %.0f x %.0f px\n",
                        det.box.x, det.box.y, det.box.width, det.box.height);
                    
                    // Afficher l'analyse des couleurs
                    if (det.colorAnalysis != null) {
                        System.out.println("     🎨 Analyse des couleurs:");
                        System.out.printf("        Couleur dominante: %s\n", det.colorAnalysis.dominantColor);
                        System.out.printf("        BGR moyen: (%.0f, %.0f, %.0f)\n", 
                            det.colorAnalysis.avgBGR[0], 
                            det.colorAnalysis.avgBGR[1], 
                            det.colorAnalysis.avgBGR[2]);
                        System.out.printf("        HSV moyen: (%.0f, %.0f, %.0f)\n", 
                            det.colorAnalysis.avgHSV[0], 
                            det.colorAnalysis.avgHSV[1], 
                            det.colorAnalysis.avgHSV[2]);
                        System.out.println("        Proportions:");
                        for (java.util.Map.Entry<String, Double> entry : det.colorAnalysis.colorProportions.entrySet()) {
                            System.out.printf("          - %s: %.1f%%\n", entry.getKey(), entry.getValue());
                        }
                    }
                    System.out.println();
                }
            }
            
            // Sauvegarder l'image avec les détections
            Mat result = detector.drawDetections(image, detections);
            String outputPath = "detection_result.jpg";
            Imgcodecs.imwrite(outputPath, result);
            
            // Extraire et sauvegarder chaque bonbon détecté séparément
            if (!detections.isEmpty()) {
                System.out.println("\n✂️ Extraction des objets détectés...");
                List<String> extractedPaths = detector.extractAllDetections(image, detections, "candy_extracted");
                System.out.println("✅ " + extractedPaths.size() + " objet(s) extrait(s):");
                for (String path : extractedPaths) {
                    System.out.println("   - " + path);
                }
            }
            
            System.out.println("\n=".repeat(60));
            System.out.println("💾 Résultats sauvegardés:");
            System.out.println("   - " + outputPath + " (image avec détections)");
            System.out.println("=".repeat(60));
            
        } catch (IOException e) {
            System.err.println("\n❌ Erreur: " + e.getMessage());
            e.printStackTrace();
        } finally {
            scanner.close();
        }
    }
}
