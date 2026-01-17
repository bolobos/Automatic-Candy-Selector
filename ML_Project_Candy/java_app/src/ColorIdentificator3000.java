import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import static org.opencv.highgui.HighGui.*;
import java.util.ArrayList;
import java.util.List;

class ColorIdentificator3000 {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    
    static Mat image;
    static Mat imgray;
    static Mat hsvImage;
    
    /**
     * Détection automatique du bonbon avec plusieurs méthodes combinées
     * Retourne le plus grand contour trouvé
     */
    public static class DetectionResult {
        MatOfPoint contour;
        Mat binaryImage;
        String methodName;
        
        DetectionResult(MatOfPoint c, Mat img, String name) {
            contour = c;
            binaryImage = img;
            methodName = name;
        }
    }
    
    public static DetectionResult detectCandyAutomatically(Mat img) {
        System.out.println("\n=== Détection Automatique du Bonbon ===");
        
        // Calculer les limites d'aire raisonnables pour un bonbon
        double totalImageArea = img.rows() * img.cols();
        double minCandyArea = totalImageArea * 0.001; // Au moins 0.1% de l'image
        double maxCandyArea = totalImageArea * 0.8;   // Max 80% de l'image (bonbon peut être gros)
        
        System.out.println("Aire image: " + (int)totalImageArea + " pixels");
        System.out.println("Aire bonbon attendue: " + (int)minCandyArea + " - " + (int)maxCandyArea + " pixels");
        
        // Méthode 1: Otsu simple
        Mat otsuBinary = new Mat();
        Imgproc.threshold(imgray, otsuBinary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        
        // Méthode 2: Otsu inversé
        Mat otsuInv = new Mat();
        Core.bitwise_not(otsuBinary, otsuInv);
        
        // Nettoyer le bruit avec morphologie simple
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(10, 10));
        Mat cleaned1 = new Mat();
        Mat cleaned2 = new Mat();
        Imgproc.morphologyEx(otsuBinary, cleaned1, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(otsuInv, cleaned2, Imgproc.MORPH_OPEN, kernel);
        
        // Chercher le meilleur contour
        MatOfPoint bestContour = null;
        Mat bestBinaryImage = null;
        double bestScore = 0;
        String bestMethod = "";
        
        Mat[] binaries = {cleaned1, cleaned2};
        String[] methodNames = {"Otsu inversé", "Otsu normal"};
        
        for (int i = 0; i < binaries.length; i++) {
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(binaries[i].clone(), contours, hierarchy, 
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            
            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                
                // Filtrer par aire raisonnable
                if (area >= minCandyArea && area <= maxCandyArea) {
                    // Calculer le ratio aire/bbox (compacité)
                    Rect bbox = Imgproc.boundingRect(contour);
                    double bboxArea = bbox.width * bbox.height;
                    double fillRatio = area / bboxArea;
                    
                    // Un bonbon compact a un fillRatio > 0.5
                    // Le fond irrégulier a un fillRatio faible
                    if (fillRatio > 0.4) {
                        // Calculer le pourcentage de l'image occupé
                        double areaPercent = area / totalImageArea;
                        
                        // Pénaliser les très grands contours (probablement le fond)
                        // Favoriser les contours entre 1% et 40% de l'image
                        double sizePenalty = 1.0;
                        if (areaPercent > 0.4) {
                            // Contour trop grand, probablement le fond
                            sizePenalty = 0.4 / areaPercent; // Pénalité forte
                        } else if (areaPercent < 0.05) {
                            // Contour un peu petit mais acceptable
                            sizePenalty = areaPercent / 0.05;
                        }
                        
                        double score = area * fillRatio * sizePenalty;
                        
                        if (score > bestScore) {
                            bestScore = score;
                            bestContour = contour;
                            bestBinaryImage = binaries[i].clone();
                            bestMethod = methodNames[i] + " (compacité: " + String.format("%.2f", fillRatio) + 
                                        ", taille: " + String.format("%.1f", areaPercent * 100) + "%)";
                        }
                    }
                }
            }
            hierarchy.release();
        }
        
        if (bestContour != null) {
            System.out.println("✓ Bonbon détecté avec méthode: " + bestMethod);
            System.out.println("  Aire: " + (int)Imgproc.contourArea(bestContour) + " pixels");
        } else {
            System.out.println("✗ Aucun bonbon détecté");
        }
        
        // Libérer la mémoire
        otsuBinary.release();
        otsuInv.release();
        cleaned1.release();
        cleaned2.release();
        kernel.release();
        
        return new DetectionResult(bestContour, bestBinaryImage, bestMethod);
    }
    
    public static MatOfPoint findLargestClosedContour(Mat edges) {
        // Trouver tous les contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        
        // Trouver le plus grand contour fermé
        MatOfPoint largestContour = null;
        double maxArea = 0;
        
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea && area > 500) { // Minimum 500 pixels (réduit de 1000)
                maxArea = area;
                largestContour = contour;
            }
        }
        
        hierarchy.release();
        return largestContour;
    }
    
    public static void analyzeCandy() {
        System.out.println("\n=== Analyse du Bonbon ===");
        
        // Détection automatique
        DetectionResult result = detectCandyAutomatically(image);
        MatOfPoint largestContour = result.contour;
        Mat binaryImage = result.binaryImage;
        
        int centerX = -1;
        int centerY = -1;
        
        // Si un contour est trouvé, calculer son centre avec les moments
        if (largestContour != null) {
            org.opencv.imgproc.Moments moments = Imgproc.moments(largestContour);
            if (moments.get_m00() != 0) {
                centerX = (int)(moments.get_m10() / moments.get_m00());
                centerY = (int)(moments.get_m01() / moments.get_m00());
            }
        }
        
        // Créer une image pour visualisation
        Mat resultImage = image.clone();
        Mat colorSample = new Mat(300, 400, CvType.CV_8UC3, new Scalar(128, 128, 128));
        
        // Dessiner le contour trouvé
        if (largestContour != null) {
            List<MatOfPoint> contours = new ArrayList<>();
            contours.add(largestContour);
            Imgproc.drawContours(resultImage, contours, 0, new Scalar(0, 255, 0), 3);
            
            // Dessiner un rectangle englobant
            Rect boundingRect = Imgproc.boundingRect(largestContour);
            Imgproc.rectangle(resultImage, boundingRect.tl(), boundingRect.br(), 
                new Scalar(255, 0, 0), 2);
        }
        
        if (centerX != -1 && centerY != -1) {
            // Calculer la couleur moyenne dans un rayon de 20 pixels autour du centre
            int radius = 20;
            double sumB = 0, sumG = 0, sumR = 0;
            double sumH = 0, sumS = 0, sumV = 0;
            int pixelCount = 0;
            
            for (int y = Math.max(0, centerY - radius); y <= Math.min(image.rows() - 1, centerY + radius); y++) {
                for (int x = Math.max(0, centerX - radius); x <= Math.min(image.cols() - 1, centerX + radius); x++) {
                    int dx = x - centerX;
                    int dy = y - centerY;
                    if (dx*dx + dy*dy <= radius*radius) {
                        double[] bgr = image.get(y, x);
                        double[] hsv = hsvImage.get(y, x);
                        sumB += bgr[0];
                        sumG += bgr[1];
                        sumR += bgr[2];
                        sumH += hsv[0];
                        sumS += hsv[1];
                        sumV += hsv[2];
                        pixelCount++;
                    }
                }
            }
            
            // Calculer les moyennes
            double[] pixelColor = new double[] {sumB / pixelCount, sumG / pixelCount, sumR / pixelCount};
            double[] pixelHSV = new double[] {sumH / pixelCount, sumS / pixelCount, sumV / pixelCount};
            
            // Dessiner le centre
            Imgproc.circle(resultImage, new Point(centerX, centerY), 8, new Scalar(0, 0, 255), -1);
            Imgproc.circle(resultImage, new Point(centerX, centerY), 12, new Scalar(255, 255, 255), 3);
            Imgproc.circle(resultImage, new Point(centerX, centerY), radius, new Scalar(255, 165, 0), 2);
            
            // Créer échantillon de couleur
            colorSample = new Mat(300, 400, CvType.CV_8UC3, 
                new Scalar(pixelColor[0], pixelColor[1], pixelColor[2]));
            
            Imgproc.putText(colorSample, "DETECTION AUTOMATIQUE", 
                new Point(20, 40), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
            Imgproc.putText(colorSample, "Centre: (" + centerX + ", " + centerY + ")", 
                new Point(20, 80), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 1);
            Imgproc.putText(colorSample, "BGR: (" + (int)pixelColor[2] + ", " + 
                (int)pixelColor[1] + ", " + (int)pixelColor[0] + ")", 
                new Point(20, 120), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 1);
            Imgproc.putText(colorSample, "HSV: (" + (int)pixelHSV[0] + ", " + 
                (int)pixelHSV[1] + ", " + (int)pixelHSV[2] + ")", 
                new Point(20, 160), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 1);
            
            // Déterminer la couleur du bonbon
            String colorName = determineColorName(pixelHSV);
            Imgproc.putText(colorSample, "Couleur: " + colorName, 
                new Point(20, 210), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
            
            // Afficher dans la console
            System.out.println("\n=== Résultats ===");
            System.out.println("Centre: (" + centerX + ", " + centerY + ")");
            System.out.println("BGR: B=" + (int)pixelColor[0] + ", G=" + (int)pixelColor[1] + ", R=" + (int)pixelColor[2]);
            System.out.println("HSV: H=" + (int)pixelHSV[0] + ", S=" + (int)pixelHSV[1] + ", V=" + (int)pixelHSV[2]);
            System.out.println("Couleur identifiée: " + colorName);
            
        } else {
            System.out.println("✗ Impossible de trouver le centre du bonbon");
            Imgproc.putText(colorSample, "Aucun bonbon detecte", 
                new Point(50, 150), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(255, 255, 255), 2);
        }
        
        // Créer une image pour visualiser les contours (sur fond blanc pour mieux voir)
        Mat contoursViz = new Mat(image.rows(), image.cols(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        if (largestContour != null) {
            List<MatOfPoint> contours = new ArrayList<>();
            contours.add(largestContour);
            // Dessiner le contour rempli en vert
            Imgproc.drawContours(contoursViz, contours, 0, new Scalar(0, 255, 0), -1);
            // Dessiner le contour en rouge épais
            Imgproc.drawContours(contoursViz, contours, 0, new Scalar(0, 0, 255), 5);
        }
        
        // Convertir l'image binaire en couleur pour l'affichage
        Mat binaryColor = new Mat();
        if (binaryImage != null) {
            Imgproc.cvtColor(binaryImage, binaryColor, Imgproc.COLOR_GRAY2BGR);
        } else {
            binaryColor = new Mat(image.rows(), image.cols(), CvType.CV_8UC3, new Scalar(0, 0, 0));
        }
        
        // Redimensionner pour l'affichage
        Mat resultResized = new Mat();
        Mat binaryResized = new Mat();
        Size displaySize = new Size(Math.max(800, image.cols() / 3.0), Math.max(600, image.rows() / 3.0));
        Imgproc.resize(resultImage, resultResized, displaySize);
        Imgproc.resize(binaryColor, binaryResized, displaySize);
        
        // Sauvegarder les résultats
        String outputPath1 = "output_detection.jpg";
        String outputPath2 = "output_couleur.jpg";
        String outputPath3 = "output_binaire.jpg";
        Imgcodecs.imwrite(outputPath1, resultResized);
        Imgcodecs.imwrite(outputPath2, colorSample);
        Imgcodecs.imwrite(outputPath3, binaryResized);
        
        System.out.println("\n✓ Images sauvegardées:");
        System.out.println("  - " + outputPath1 + " (détection avec rectangle et centre)");
        System.out.println("  - " + outputPath2 + " (échantillon de couleur)");
        System.out.println("  - " + outputPath3 + " (image binaire utilisée pour la détection)");
        
        // Afficher les résultats dans des fenêtres
        try {
            HighGui.imshow("Detection Automatique", resultResized);
            HighGui.imshow("Couleur du Bonbon", colorSample);
            HighGui.imshow("Image Binaire", binaryResized);
            
            System.out.println("\nFenêtres affichées. Appuyez sur une touche pour fermer...");
            HighGui.waitKey(0);
            HighGui.destroyAllWindows();
        } catch (Exception e) {
            System.out.println("\n⚠ Impossible d'afficher les fenêtres (mode headless?)");
            System.out.println("  Consultez les images sauvegardées ci-dessus.");
        }
        
        resultImage.release();
        resultResized.release();
    }
    
    /**
     * Détermine le nom de la couleur à partir des valeurs HSV
     */
    public static String determineColorName(double[] hsv) {
        double h = hsv[0];
        double s = hsv[1];
        double v = hsv[2];
        
        // Si saturation très faible, c'est du gris/blanc/noir
        if (s < 30) {
            if (v > 200) return "Blanc";
            else if (v < 50) return "Noir";
            else return "Gris";
        }
        
        // Déterminer la couleur basée sur la teinte (Hue)
        if (h < 10 || h > 160) return "Rouge";
        else if (h < 25) return "Orange";
        else if (h < 35) return "Jaune";
        else if (h < 85) return "Vert";
        else if (h < 130) return "Bleu";
        else return "Violet/Rose";
    }
    
    public static void main(String[] args) {
        System.out.println("Welcome to OpenCV " + Core.VERSION);
        System.out.println("=== ColorIdentificator3000 - Mode Automatique ===\n");
        
        // Charger l'image
        String imagePath = "nos_dataset/Entrainement/Dragibus/20260107_192537.jpg";
        image = Imgcodecs.imread(imagePath);
        
        if (image.empty()) {
            System.out.println("Erreur: Impossible de charger l'image: " + imagePath);
            return;
        }
        
        System.out.println("✓ Image chargée avec succès!");
        System.out.println("  Dimensions: " + image.cols() + "x" + image.rows());
        
        // Convertir en niveaux de gris
        imgray = new Mat();
        Imgproc.cvtColor(image, imgray, Imgproc.COLOR_BGR2GRAY);
        
        // Convertir en HSV
        hsvImage = new Mat();
        Imgproc.cvtColor(image, hsvImage, Imgproc.COLOR_BGR2HSV);
        
        // Analyse automatique
        analyzeCandy();
        
        System.out.println("\n=== Terminé ===");
        System.exit(0);
    }
}
