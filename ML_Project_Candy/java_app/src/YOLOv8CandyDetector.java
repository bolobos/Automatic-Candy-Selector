import org.opencv.core.*;
import org.opencv.dnn.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Détecteur de bonbons utilisant YOLOv8 en ONNX avec OpenCV DNN
 */
public class YOLOv8CandyDetector {
    
    private Net net;
    private List<String> classNames;
    private static final int INPUT_SIZE = 640;
    private static final float CONF_THRESHOLD = 0.5f;
    private static final float NMS_THRESHOLD = 0.4f;
    
    /**
     * Constructeur - charge le modèle ONNX et les noms de classes
     */
    public YOLOv8CandyDetector(String modelPath, String classNamesPath) throws IOException {
        // Charger le modèle ONNX
        this.net = Dnn.readNetFromONNX(modelPath);
        
        // Configuration pour utiliser le CPU (ou GPU si disponible)
        this.net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
        this.net.setPreferableTarget(Dnn.DNN_TARGET_CPU);
        
        // Charger les noms des classes
        this.classNames = Files.readAllLines(Paths.get(classNamesPath));
        
        System.out.println("✅ Modèle YOLOv8 chargé: " + modelPath);
        System.out.println("📝 Classes: " + String.join(", ", classNames));
    }
    
    /**
     * Détecte les bonbons dans une image
     * @param imagePath Chemin vers l'image
     * @return Liste des détections
     */
    public List<Detection> detect(String imagePath) {
        // Charger l'image
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            throw new IllegalArgumentException("Impossible de charger l'image: " + imagePath);
        }
        
        return detect(image);
    }
    
    /**
     * Détecte les bonbons dans une Mat OpenCV
     */
    public List<Detection> detect(Mat image) {
        // Dimensions originales
        int origWidth = image.cols();
        int origHeight = image.rows();
        
        // Préparer le blob (normalisation 0-1, redimensionnement 640x640)
        Mat blob = Dnn.blobFromImage(
            image,
            1.0 / 255.0,           // Scale factor (normalisation)
            new Size(INPUT_SIZE, INPUT_SIZE),  // Taille d'entrée
            new Scalar(0, 0, 0),   // Mean subtraction (aucune)
            true,                  // Swap RB (BGR -> RGB)
            false                  // Crop
        );
        
        // Inférence
        net.setInput(blob);
        Mat output = net.forward();
        
        // Post-traitement YOLOv8
        return postProcess(output, origWidth, origHeight);
    }
    
    /**
     * Post-traitement des sorties YOLOv8
     * Format: [1, 84, 8400] où 84 = 4 (bbox) + 80 (classes)
     */
    private List<Detection> postProcess(Mat output, int origWidth, int origHeight) {
        List<Detection> detections = new ArrayList<>();
        
        // Transposer la sortie: [1, 84, 8400] -> [8400, 84]
        Mat output2D = output.reshape(1, output.size(1));  // [84, 8400]
        Core.transpose(output2D, output2D);                // [8400, 84]
        
        // Facteurs d'échelle pour revenir aux dimensions originales
        float xScale = (float) origWidth / INPUT_SIZE;
        float yScale = (float) origHeight / INPUT_SIZE;
        
        // Listes temporaires pour NMS
        List<Rect2d> boxes = new ArrayList<>();
        List<Float> confidences = new ArrayList<>();
        List<Integer> classIds = new ArrayList<>();
        
        // Parcourir les 8400 prédictions
        for (int i = 0; i < output2D.rows(); i++) {
            Mat row = output2D.row(i);
            
            // Extraire les coordonnées bbox (cx, cy, w, h)
            float cx = (float) row.get(0, 0)[0];
            float cy = (float) row.get(0, 1)[0];
            float w = (float) row.get(0, 2)[0];
            float h = (float) row.get(0, 3)[0];
            
            // Trouver la classe avec la confidence maximale (colonnes 4-83)
            float maxConf = 0;
            int maxClassId = -1;
            
            for (int c = 4; c < Math.min(4 + classNames.size(), 84); c++) {
                float conf = (float) row.get(0, c)[0];
                if (conf > maxConf) {
                    maxConf = conf;
                    maxClassId = c - 4;
                }
            }
            
            // Filtrer par seuil de confiance
            if (maxConf >= CONF_THRESHOLD) {
                // Convertir (cx, cy, w, h) -> (x, y, width, height)
                float x = (cx - w / 2) * xScale;
                float y = (cy - h / 2) * yScale;
                float width = w * xScale;
                float height = h * yScale;
                
                boxes.add(new Rect2d(x, y, width, height));
                confidences.add(maxConf);
                classIds.add(maxClassId);
            }
        }
        
        // Appliquer NMS (Non-Maximum Suppression)
        MatOfRect2d boxesMat = new MatOfRect2d();
        boxesMat.fromList(boxes);
        
        MatOfFloat confMat = new MatOfFloat();
        float[] confArray = new float[confidences.size()];
        for (int i = 0; i < confidences.size(); i++) {
            confArray[i] = confidences.get(i);
        }
        confMat.fromArray(confArray);
        
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxesMat, confMat, CONF_THRESHOLD, NMS_THRESHOLD, indices);
        
        // Créer les détections finales
        int[] indicesArray = indices.toArray();
        for (int idx : indicesArray) {
            Rect2d box = boxes.get(idx);
            detections.add(new Detection(
                classNames.get(classIds.get(idx)),
                classIds.get(idx),
                confidences.get(idx),
                box
            ));
        }
        
        return detections;
    }
    
    /**
     * Classe représentant une détection
     */
    public static class Detection {
        public String className;
        public int classId;
        public float confidence;
        public Rect2d box;
        public ColorAnalysis colorAnalysis;
        
        public Detection(String className, int classId, float confidence, Rect2d box) {
            this.className = className;
            this.classId = classId;
            this.confidence = confidence;
            this.box = box;
            this.colorAnalysis = null;
        }
        
        @Override
        public String toString() {
            return String.format("%s (%.2f%%) [%.0f, %.0f, %.0fx%.0f]",
                className, confidence * 100, box.x, box.y, box.width, box.height);
        }
    }
    
    /**
     * Classe représentant l'analyse des couleurs d'un objet
     */
    public static class ColorAnalysis {
        public double[] avgBGR;
        public double[] avgHSV;
        public String dominantColor;
        public java.util.Map<String, Double> colorProportions;
        
        public ColorAnalysis(double[] bgr, double[] hsv, String color, java.util.Map<String, Double> proportions) {
            this.avgBGR = bgr;
            this.avgHSV = hsv;
            this.dominantColor = color;
            this.colorProportions = proportions;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("Couleur dominante: %s\n", dominantColor));
            sb.append(String.format("BGR moyen: (%.0f, %.0f, %.0f)\n", avgBGR[0], avgBGR[1], avgBGR[2]));
            sb.append(String.format("HSV moyen: (%.0f, %.0f, %.0f)\n", avgHSV[0], avgHSV[1], avgHSV[2]));
            sb.append("Proportions de couleurs:\n");
            for (java.util.Map.Entry<String, Double> entry : colorProportions.entrySet()) {
                sb.append(String.format("  %s: %.1f%%\n", entry.getKey(), entry.getValue()));
            }
            return sb.toString();
        }
    }
    
    /**
     * Dessine les détections sur une image
     */
    public Mat drawDetections(Mat image, List<Detection> detections) {
        Mat result = image.clone();
        
        // Couleurs pour chaque classe (BGR)
        Scalar[] colors = {
            new Scalar(0, 0, 255),    // Rouge
            new Scalar(0, 255, 0),    // Vert
            new Scalar(255, 0, 0),    // Bleu
            new Scalar(0, 255, 255),  // Jaune
            new Scalar(255, 0, 255),  // Magenta
            new Scalar(255, 255, 0)   // Cyan
        };
        
        for (Detection det : detections) {
            Scalar color = colors[det.classId % colors.length];
            
            // Rectangle
            Point p1 = new Point(det.box.x, det.box.y);
            Point p2 = new Point(det.box.x + det.box.width, det.box.y + det.box.height);
            Imgproc.rectangle(result, p1, p2, color, 2);
            
            // Label
            String label = String.format("%s %.1f%%", det.className, det.confidence * 100);
            Point textPos = new Point(det.box.x, det.box.y - 5);
            Imgproc.putText(result, label, textPos, 
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
        
        return result;
    }
    
    /**
     * Analyse les couleurs d'un objet détecté dans l'image
     */
    public ColorAnalysis analyzeColors(Mat image, Detection detection) {
        // Extraire la région d'intérêt (ROI)
        Rect roi = new Rect(
            (int)Math.max(0, detection.box.x),
            (int)Math.max(0, detection.box.y),
            (int)Math.min(detection.box.width, image.cols() - detection.box.x),
            (int)Math.min(detection.box.height, image.rows() - detection.box.y)
        );
        
        Mat candyROI = new Mat(image, roi);
        
        // Convertir en HSV pour l'analyse des couleurs
        Mat hsvROI = new Mat();
        Imgproc.cvtColor(candyROI, hsvROI, Imgproc.COLOR_BGR2HSV);
        
        // Calculer les moyennes BGR et HSV
        double sumB = 0, sumG = 0, sumR = 0;
        double sumH = 0, sumS = 0, sumV = 0;
        int pixelCount = 0;
        
        // Compteur pour chaque catégorie de couleur
        java.util.Map<String, Integer> colorCounts = new java.util.HashMap<>();
        colorCounts.put("Rouge", 0);
        colorCounts.put("Orange", 0);
        colorCounts.put("Jaune", 0);
        colorCounts.put("Vert", 0);
        colorCounts.put("Bleu", 0);
        colorCounts.put("Violet/Rose", 0);
        colorCounts.put("Blanc", 0);
        colorCounts.put("Gris", 0);
        colorCounts.put("Noir", 0);
        
        // Parcourir tous les pixels de la ROI
        for (int y = 0; y < candyROI.rows(); y++) {
            for (int x = 0; x < candyROI.cols(); x++) {
                double[] bgr = candyROI.get(y, x);
                double[] hsv = hsvROI.get(y, x);
                
                sumB += bgr[0];
                sumG += bgr[1];
                sumR += bgr[2];
                sumH += hsv[0];
                sumS += hsv[1];
                sumV += hsv[2];
                
                // Déterminer la couleur de ce pixel
                String pixelColor = determineColorName(hsv);
                colorCounts.put(pixelColor, colorCounts.get(pixelColor) + 1);
                
                pixelCount++;
            }
        }
        
        // Calculer les moyennes
        double[] avgBGR = new double[] {
            sumB / pixelCount,
            sumG / pixelCount,
            sumR / pixelCount
        };
        
        double[] avgHSV = new double[] {
            sumH / pixelCount,
            sumS / pixelCount,
            sumV / pixelCount
        };
        
        // Déterminer la couleur dominante
        String dominantColor = determineColorName(avgHSV);
        
        // Calculer les proportions de chaque couleur en pourcentage
        java.util.Map<String, Double> colorProportions = new java.util.LinkedHashMap<>();
        for (java.util.Map.Entry<String, Integer> entry : colorCounts.entrySet()) {
            double proportion = (entry.getValue() * 100.0) / pixelCount;
            if (proportion > 0.5) { // Ne garder que les couleurs représentant au moins 0.5%
                colorProportions.put(entry.getKey(), proportion);
            }
        }
        
        // Trier les proportions par valeur décroissante
        java.util.List<java.util.Map.Entry<String, Double>> sortedProportions = 
            new java.util.ArrayList<>(colorProportions.entrySet());
        sortedProportions.sort((e1, e2) -> e2.getValue().compareTo(e1.getValue()));
        
        java.util.Map<String, Double> sortedColorProportions = new java.util.LinkedHashMap<>();
        for (java.util.Map.Entry<String, Double> entry : sortedProportions) {
            sortedColorProportions.put(entry.getKey(), entry.getValue());
        }
        
        hsvROI.release();
        
        return new ColorAnalysis(avgBGR, avgHSV, dominantColor, sortedColorProportions);
    }
    
    /**
     * Détermine le nom de la couleur à partir des valeurs HSV
     */
    private String determineColorName(double[] hsv) {
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
    
    /**
     * Analyse les couleurs de tous les objets détectés
     */
    public void analyzeColorsForDetections(Mat image, List<Detection> detections) {
        for (Detection det : detections) {
            det.colorAnalysis = analyzeColors(image, det);
        }
    }
    
    /**
     * Extrait l'objet détecté et le sauvegarde comme image séparée avec fond transparent
     * @param image Image source
     * @param detection Détection à extraire
     * @param outputPath Chemin du fichier de sortie
     */
    public void extractAndSaveDetection(Mat image, Detection detection, String outputPath) {
        // Créer un rectangle sûr dans les limites de l'image
        Rect roi = new Rect(
            (int)Math.max(0, detection.box.x),
            (int)Math.max(0, detection.box.y),
            (int)Math.min(detection.box.width, image.cols() - detection.box.x),
            (int)Math.min(detection.box.height, image.rows() - detection.box.y)
        );
        
        // Extraire la région d'intérêt
        Mat croppedObject = new Mat(image, roi);
        
        // Créer un masque intelligent pour isoler l'objet du fond
        Mat mask = createSmartMask(croppedObject);
        
        // Créer une image avec canal alpha (BGRA)
        Mat result = new Mat();
        List<Mat> channels = new ArrayList<>();
        Core.split(croppedObject, channels);
        
        // Ajouter le masque comme canal alpha
        channels.add(mask);
        Core.merge(channels, result);
        
        // Sauvegarder l'image avec transparence
        Imgcodecs.imwrite(outputPath, result);
        
        // Libérer la mémoire
        mask.release();
        for (Mat ch : channels) {
            ch.release();
        }
        result.release();
    }
    
    /**
     * Crée un masque intelligent qui préserve l'objet tout en supprimant le fond
     * Utilise une approche multi-critères : bords + couleur
     */
    private Mat createSmartMask(Mat roi) {
        // Convertir en HSV pour une meilleure détection des couleurs
        Mat hsv = new Mat();
        Imgproc.cvtColor(roi, hsv, Imgproc.COLOR_BGR2HSV);
        
        // Séparer les canaux HSV
        List<Mat> hsvChannels = new ArrayList<>();
        Core.split(hsv, hsvChannels);
        Mat saturation = hsvChannels.get(1); // Canal S (saturation)
        Mat value = hsvChannels.get(2);      // Canal V (luminosité)
        
        // 1. Créer un masque basé sur la saturation
        // Les bonbons ont généralement une saturation élevée (>40)
        // Le fond blanc et les ombres ont une faible saturation
        Mat satMask = new Mat();
        Imgproc.threshold(saturation, satMask, 40, 255, Imgproc.THRESH_BINARY);
        
        // 2. Enlever les zones trop claires (fond blanc) ET trop sombres (ombres foncées)
        Mat valueMask = new Mat();
        Mat valueHigh = new Mat();
        Mat valueLow = new Mat();
        Imgproc.threshold(value, valueHigh, 210, 255, Imgproc.THRESH_BINARY_INV); // Enlever blanc
        Imgproc.threshold(value, valueLow, 50, 255, Imgproc.THRESH_BINARY);      // Enlever noir/ombres très foncées
        Core.bitwise_and(valueHigh, valueLow, valueMask);
        
        // 3. Combiner les deux masques
        Mat combinedMask = new Mat();
        Core.bitwise_and(satMask, valueMask, combinedMask);
        
        // 4. Trouver le plus grand contour (l'objet principal)
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(combinedMask.clone(), contours, hierarchy, 
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        
        // Créer un masque avec seulement le plus grand contour
        Mat finalMask = Mat.zeros(roi.size(), CvType.CV_8UC1);
        if (!contours.isEmpty()) {
            // Trouver le plus grand contour
            MatOfPoint largestContour = null;
            double maxArea = 0;
            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                if (area > maxArea) {
                    maxArea = area;
                    largestContour = contour;
                }
            }
            
            // Dessiner le contour rempli
            if (largestContour != null) {
                List<MatOfPoint> largestContourList = new ArrayList<>();
                largestContourList.add(largestContour);
                Imgproc.drawContours(finalMask, largestContourList, 0, new Scalar(255), -1);
            }
        }
        
        // 5. Fermer les petits trous dans l'objet
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.morphologyEx(finalMask, finalMask, Imgproc.MORPH_CLOSE, kernel);
        
        // 6. Dilater légèrement pour récupérer les bords
        Mat kernelDilate = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        Imgproc.dilate(finalMask, finalMask, kernelDilate, new Point(-1, -1), 3);
        
        // 7. Adoucir les bords
        Mat smoothMask = new Mat();
        Imgproc.GaussianBlur(finalMask, smoothMask, new Size(5, 5), 0);
        
        // Libérer mémoire
        hsv.release();
        for (Mat ch : hsvChannels) {
            ch.release();
        }
        saturation.release();
        value.release();
        satMask.release();
        valueHigh.release();
        valueLow.release();
        valueMask.release();
        combinedMask.release();
        hierarchy.release();
        kernel.release();
        kernelDilate.release();
        finalMask.release();
        
        return smoothMask;
    }
    
    /**
     * Crée un masque pour isoler l'objet du fond
     * Utilise plusieurs méthodes de segmentation pour trouver le meilleur masque
     */
    private Mat createObjectMask(Mat roi) {
        // Convertir en niveaux de gris
        Mat gray = new Mat();
        Imgproc.cvtColor(roi, gray, Imgproc.COLOR_BGR2GRAY);
        
        // Méthode 1: Otsu inversé
        Mat otsuBinary = new Mat();
        Imgproc.threshold(gray, otsuBinary, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        
        // Méthode 2: Otsu normal
        Mat otsuInv = new Mat();
        Core.bitwise_not(otsuBinary, otsuInv);
        
        // Nettoyer le bruit avec morphologie LÉGÈRE (kernel plus petit)
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
        Mat cleaned1 = new Mat();
        Mat cleaned2 = new Mat();
        // Seulement CLOSE pour combler les trous, pas d'OPEN pour éviter d'éroder
        Imgproc.morphologyEx(otsuBinary, cleaned1, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(otsuInv, cleaned2, Imgproc.MORPH_CLOSE, kernel);
        
        // Trouver le meilleur masque en cherchant le plus grand contour compact
        Mat bestMask = selectBestMask(cleaned1, cleaned2, roi);
        
        // Dilater légèrement pour récupérer les bords
        Mat kernelDilate = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2, 2));
        Mat dilatedMask = new Mat();
        Imgproc.dilate(bestMask, dilatedMask, kernelDilate);
        
        // Appliquer un flou gaussien très léger pour adoucir les bords
        Mat smoothMask = new Mat();
        Imgproc.GaussianBlur(dilatedMask, smoothMask, new Size(3, 3), 0);
        
        // Libérer la mémoire
        gray.release();
        otsuBinary.release();
        otsuInv.release();
        kernel.release();
        kernelDilate.release();
        cleaned1.release();
        cleaned2.release();
        bestMask.release();
        dilatedMask.release();
        
        return smoothMask;
    }
    
    /**
     * Sélectionne le meilleur masque parmi les candidats
     */
    private Mat selectBestMask(Mat mask1, Mat mask2, Mat roi) {
        double totalArea = roi.rows() * roi.cols();
        Mat[] masks = {mask1, mask2};
        Mat bestMask = null;
        double bestScore = 0;
        
        for (Mat mask : masks) {
            // Trouver les contours
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(mask.clone(), contours, hierarchy, 
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            
            for (MatOfPoint contour : contours) {
                double area = Imgproc.contourArea(contour);
                
                // Filtrer par aire raisonnable (entre 10% et 98% de l'image) - plus permissif
                if (area >= totalArea * 0.10 && area <= totalArea * 0.98) {
                    // Calculer la compacité
                    Rect bbox = Imgproc.boundingRect(contour);
                    double bboxArea = bbox.width * bbox.height;
                    double fillRatio = area / bboxArea;
                    
                    // Favoriser les contours compacts, mais être plus tolérant (seuil réduit à 0.2)
                    if (fillRatio > 0.2) {
                        double score = area * fillRatio;
                        
                        if (score > bestScore) {
                            bestScore = score;
                            // Créer un masque avec ce contour
                            Mat tempMask = Mat.zeros(mask.size(), CvType.CV_8UC1);
                            List<MatOfPoint> singleContour = new ArrayList<>();
                            singleContour.add(contour);
                            Imgproc.drawContours(tempMask, singleContour, 0, new Scalar(255), -1);
                            
                            if (bestMask != null) {
                                bestMask.release();
                            }
                            bestMask = tempMask;
                        }
                    }
                }
            }
            hierarchy.release();
        }
        
        // Si aucun bon masque trouvé, utiliser le premier masque par défaut
        if (bestMask == null) {
            bestMask = mask1.clone();
        }
        
        return bestMask;
    }
    
    /**
     * Extrait tous les objets détectés et les sauvegarde
     * @param image Image source
     * @param detections Liste des détections
     * @param outputPrefix Préfixe pour les noms de fichiers
     * @return Liste des chemins des fichiers créés
     */
    public List<String> extractAllDetections(Mat image, List<Detection> detections, String outputPrefix) {
        List<String> savedPaths = new ArrayList<>();
        
        for (int i = 0; i < detections.size(); i++) {
            Detection det = detections.get(i);
            String outputPath = String.format("%s_%d_%s.png", 
                outputPrefix, i + 1, det.className.toLowerCase());
            extractAndSaveDetection(image, det, outputPath);
            savedPaths.add(outputPath);
        }
        
        return savedPaths;
    }
}
