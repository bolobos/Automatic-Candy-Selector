import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import static org.opencv.highgui.HighGui.*;
import java.util.ArrayList;
import java.util.List;

class ColorIdentificator3000 {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    
    // Variables globales pour les seuils Canny
    static int minThreshold = 150;
    static int maxThreshold = 300;
    static Mat image;
    static Mat imgray;
    static Mat hsvImage;
    static boolean autoMode = false;
    
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
    
    public static void autoAdjustThresholds() {
        System.out.println("\n=== Mode Auto: Recherche du bonbon ===");
        
        double bestArea = 0;
        int bestT1 = minThreshold;
        int bestT2 = maxThreshold;
        
        // Tester différents seuils pour trouver un contour fermé
        for (int t1 = 50; t1 <= 400; t1 += 25) {
            for (int t2 = t1 + 50; t2 <= 500; t2 += 25) {
                Mat edges = new Mat();
                Imgproc.Canny(imgray, edges, t1, t2);
                
                MatOfPoint largestContour = findLargestClosedContour(edges);
                
                if (largestContour != null) {
                    double area = Imgproc.contourArea(largestContour);
                    if (area > bestArea) {
                        bestArea = area;
                        bestT1 = t1;
                        bestT2 = t2;
                        System.out.println("Contour trouvé: T1=" + t1 + ", T2=" + t2 + ", Aire=" + (int)area);
                    }
                }
                edges.release();
            }
        }
        
        if (bestArea > 0) {
            minThreshold = bestT1;
            maxThreshold = bestT2;
            System.out.println("\n>>> Meilleur résultat: MinThreshold=" + minThreshold + ", MaxThreshold=" + maxThreshold + " ===");
            System.out.println(">>> Aire du contour: " + (int)bestArea + " pixels");
        } else {
            System.out.println("Aucun bonbon détecté, garde les valeurs actuelles");
        }
    }
    
    public static void updateEdges() {
        // Détection de contours avec Canny
        Mat edges = new Mat();
        Imgproc.Canny(imgray, edges, minThreshold, maxThreshold);
        
        // Chercher le plus grand contour fermé
        MatOfPoint largestContour = findLargestClosedContour(edges);
        
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
        
        // Sinon, utiliser la méthode originale (moyenne des pixels de contour)
        if (centerX == -1 || centerY == -1) {
            long sumX = 0;
            long sumY = 0;
            int edgePixelCount = 0;
            
            for (int y = 0; y < edges.rows(); y++) {
                for (int x = 0; x < edges.cols(); x++) {
                    double[] pixel = edges.get(y, x);
                    if (pixel[0] == 255) {
                        sumX += x;
                        sumY += y;
                        edgePixelCount++;
                    }
                }
            }
            
            if (edgePixelCount > 0) {
                centerX = (int)(sumX / edgePixelCount);
                centerY = (int)(sumY / edgePixelCount);
            }
        }
        
        // Redimensionner l'image pour l'affichage (réduire à 25%)
        Mat imageResized = new Mat();
        Mat edgesResized = new Mat();
        Size displaySize = new Size(image.cols() / 4.0, image.rows() / 4.0);
        Imgproc.resize(image, imageResized, displaySize);
        Imgproc.resize(edges, edgesResized, displaySize);
        
        // Dessiner le contour trouvé sur l'image redimensionnée
        if (largestContour != null) {
            List<MatOfPoint> contours = new ArrayList<>();
            MatOfPoint scaledContour = new MatOfPoint();
            Point[] points = largestContour.toArray();
            Point[] scaledPoints = new Point[points.length];
            for (int i = 0; i < points.length; i++) {
                scaledPoints[i] = new Point(points[i].x / 4.0, points[i].y / 4.0);
            }
            scaledContour.fromArray(scaledPoints);
            contours.add(scaledContour);
            Imgproc.drawContours(imageResized, contours, 0, new Scalar(0, 255, 255), 2);
        }
        
        // Ajouter les valeurs de threshold sur l'image des contours
        Imgproc.putText(edgesResized, "MinThreshold: " + minThreshold, 
                      new Point(10, 30), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
        Imgproc.putText(edgesResized, "MaxThreshold: " + maxThreshold, 
                      new Point(10, 70), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(255, 255, 255), 2);
        
        Mat imageWithCenter = imageResized.clone();
        Mat colorSample = new Mat(200, 350, CvType.CV_8UC3, new Scalar(128, 128, 128));
        
        if (centerX != -1 && centerY != -1) {
            // Calculer la couleur moyenne dans un rayon de 10 pixels autour du centre
            int radius = 10;
            double sumB = 0, sumG = 0, sumR = 0;
            double sumH = 0, sumS = 0, sumV = 0;
            int pixelCount = 0;
            
            for (int y = Math.max(0, centerY - radius); y <= Math.min(image.rows() - 1, centerY + radius); y++) {
                for (int x = Math.max(0, centerX - radius); x <= Math.min(image.cols() - 1, centerX + radius); x++) {
                    // Vérifier si le pixel est dans le cercle de rayon 10
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
            
            // Dessiner le centre (coordonnées réduites à 25%)
            int centerXResized = centerX / 4;
            int centerYResized = centerY / 4;
            Imgproc.circle(imageWithCenter, new Point(centerXResized, centerYResized), 4, new Scalar(0, 0, 255), -1);
            Imgproc.circle(imageWithCenter, new Point(centerXResized, centerYResized), 6, new Scalar(255, 255, 255), 2);
            // Dessiner le cercle de moyennage (rayon réduit à 25%)
            Imgproc.circle(imageWithCenter, new Point(centerXResized, centerYResized), radius / 4, new Scalar(255, 165, 0), 1);
            
            // Afficher les thresholds sur l'image (en bas pour être visibles)
            int imgHeight = (int)displaySize.height;
            Imgproc.putText(imageWithCenter, "MinThreshold: " + minThreshold, 
                          new Point(10, imgHeight - 70), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
            Imgproc.putText(imageWithCenter, "MaxThreshold: " + maxThreshold, 
                          new Point(10, imgHeight - 30), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
            
            // Créer échantillon de couleur avec texte
            colorSample = new Mat(200, 350, CvType.CV_8UC3, new Scalar(pixelColor[0], pixelColor[1], pixelColor[2]));
            Imgproc.putText(colorSample, "MinThreshold: " + minThreshold, 
                          new Point(10, 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 2);
            Imgproc.putText(colorSample, "MaxThreshold: " + maxThreshold, 
                          new Point(10, 60), Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 2);
            Imgproc.putText(colorSample, "BGR: " + (int)pixelColor[2] + "," + (int)pixelColor[1] + "," + (int)pixelColor[0], 
                          new Point(10, 100), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 1);
            Imgproc.putText(colorSample, "HSV: " + (int)pixelHSV[0] + "," + (int)pixelHSV[1] + "," + (int)pixelHSV[2], 
                          new Point(10, 130), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 1);
            
            // Afficher les infos dans la console
            System.out.println("\n=== MinThreshold=" + minThreshold + ", MaxThreshold=" + maxThreshold + " ===");
            System.out.println("Centre: (" + centerX + ", " + centerY + ")");
            System.out.println("BGR: B=" + (int)pixelColor[0] + ", G=" + (int)pixelColor[1] + ", R=" + (int)pixelColor[2]);
            System.out.println("HSV: H=" + (int)pixelHSV[0] + ", S=" + (int)pixelHSV[1] + ", V=" + (int)pixelHSV[2]);
        } else {
            System.out.println("\n=== MinThreshold=" + minThreshold + ", MaxThreshold=" + maxThreshold + " ===");
            System.out.println("Aucun pixel de contour trouvé!");
            Imgproc.putText(colorSample, "Aucun contour", new Point(50, 75), 
                          Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
        }
        
        // Afficher les résultats (images redimensionnées)
        HighGui.imshow("Contours (Canny)", edgesResized);
        HighGui.imshow("Centre des Contours (Suzuki-Abe)", imageWithCenter);
        HighGui.imshow("Couleur au Centre", colorSample);
    }
    
    public static void main(String[] args) {
        System.out.println("Welcome to OpenCV " + Core.VERSION);
        
        // Charger l'image
        String imagePath = "./ML_Project_Candy/nos_dataset/Entrainement/Croco/20251015_100344.jpg";
        image = Imgcodecs.imread(imagePath);
        
        if (image.empty()) {
            System.out.println("Erreur: Impossible de charger l'image: " + imagePath);
            return;
        }
        
        System.out.println("Image chargée avec succès!");
        System.out.println("Dimensions: " + image.cols() + "x" + image.rows());
        
        // Convertir en niveaux de gris
        imgray = new Mat();
        Imgproc.cvtColor(image, imgray, Imgproc.COLOR_BGR2GRAY);
        
        // Convertir en HSV
        hsvImage = new Mat();
        Imgproc.cvtColor(image, hsvImage, Imgproc.COLOR_BGR2HSV);
        
        // Créer les fenêtres
        namedWindow("Contours (Canny)");
        namedWindow("Centre des Contours (Suzuki-Abe)");
        namedWindow("Couleur au Centre");
        
        System.out.println("\n=== Interface Interactive ===");
        System.out.println("Utilisez les touches pour ajuster les seuils (modification +/-10):");
        System.out.println("  '1' / '2' : Diminuer/Augmenter MinThreshold (actuellement: " + minThreshold + ")");
        System.out.println("  '3' / '4' : Diminuer/Augmenter MaxThreshold (actuellement: " + maxThreshold + ")");
        System.out.println("  'a' : Mode Auto (détection automatique du bonbon)");
        System.out.println("  'r' : Reset (150, 300)");
        System.out.println("  'ESC' : Quitter");
        System.out.println("\nImages redimensionnées à 25% pour un meilleur affichage");
        
        // Affichage initial
        updateEdges();
        
        // Boucle interactive
        while (true) {
            int key = waitKey(0); // Attendre une touche
            
            if (key == 27 || key == -1) { // ESC
                break;
            } else if (key == 'a' || key == 'A') {
                autoAdjustThresholds();
                updateEdges();
            } else if (key == '1') {
                minThreshold = Math.max(0, minThreshold - 10);
                System.out.println("MinThreshold: " + minThreshold);
                updateEdges();
            } else if (key == '2') {
                minThreshold = Math.min(500, minThreshold + 10);
                System.out.println("MinThreshold: " + minThreshold);
                updateEdges();
            } else if (key == '3') {
                maxThreshold = Math.max(0, maxThreshold - 10);
                System.out.println("MaxThreshold: " + maxThreshold);
                updateEdges();
            } else if (key == '4') {
                maxThreshold = Math.min(500, maxThreshold + 10);
                System.out.println("MaxThreshold: " + maxThreshold);
                updateEdges();
            } else if (key == 'r' || key == 'R') {
                minThreshold = 150;
                maxThreshold = 300;
                System.out.println("Reset: MinThreshold=" + minThreshold + ", MaxThreshold=" + maxThreshold);
                updateEdges();
            }
        }
        
        destroyAllWindows();
        System.out.println("\nProgramme terminé!");
        System.exit(0);
    }
}
