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
        
        public Detection(String className, int classId, float confidence, Rect2d box) {
            this.className = className;
            this.classId = classId;
            this.confidence = confidence;
            this.box = box;
        }
        
        @Override
        public String toString() {
            return String.format("%s (%.2f%%) [%.0f, %.0f, %.0fx%.0f]",
                className, confidence * 100, box.x, box.y, box.width, box.height);
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
}
