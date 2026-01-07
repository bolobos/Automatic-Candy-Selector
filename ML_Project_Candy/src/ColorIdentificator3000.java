import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import static org.opencv.highgui.HighGui.*;

class ColorIdentificator3000 {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    
    // Variables globales pour les seuils Canny
    static int threshold1 = 100;
    static int threshold2 = 200;
    static Mat image;
    static Mat imgray;
    static Mat hsvImage;
    
    public static void updateEdges() {
        // Détection de contours avec Canny
        Mat edges = new Mat();
        Imgproc.Canny(imgray, edges, threshold1, threshold2);
        
        // Calculer la moyenne des positions des pixels de contour
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
        
        // Redimensionner l'image pour l'affichage (réduire à 25%)
        Mat imageResized = new Mat();
        Mat edgesResized = new Mat();
        Size displaySize = new Size(image.cols() / 4.0, image.rows() / 4.0);
        Imgproc.resize(image, imageResized, displaySize);
        Imgproc.resize(edges, edgesResized, displaySize);
        
        Mat imageWithCenter = imageResized.clone();
        Mat colorSample = new Mat(150, 300, CvType.CV_8UC3, new Scalar(128, 128, 128));
        
        if (edgePixelCount > 0) {
            // Calculer la moyenne
            int centerX = (int)(sumX / edgePixelCount);
            int centerY = (int)(sumY / edgePixelCount);
            
            // Extraire la couleur du pixel au centre (coordonnées originales)
            double[] pixelColor = image.get(centerY, centerX);
            double[] pixelHSV = hsvImage.get(centerY, centerX);
            
            // Dessiner le centre (coordonnées réduites à 25%)
            int centerXResized = centerX / 4;
            int centerYResized = centerY / 4;
            Imgproc.circle(imageWithCenter, new Point(centerXResized, centerYResized), 4, new Scalar(0, 0, 255), -1);
            Imgproc.circle(imageWithCenter, new Point(centerXResized, centerYResized), 6, new Scalar(255, 255, 255), 2);
            
            // Créer échantillon de couleur avec texte
            colorSample = new Mat(150, 300, CvType.CV_8UC3, new Scalar(pixelColor[0], pixelColor[1], pixelColor[2]));
            Imgproc.putText(colorSample, "BGR: " + (int)pixelColor[2] + "," + (int)pixelColor[1] + "," + (int)pixelColor[0], 
                          new Point(10, 30), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 1);
            Imgproc.putText(colorSample, "HSV: " + (int)pixelHSV[0] + "," + (int)pixelHSV[1] + "," + (int)pixelHSV[2], 
                          new Point(10, 60), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 1);
            
            // Afficher les infos dans la console
            System.out.println("\n=== Threshold1=" + threshold1 + ", Threshold2=" + threshold2 + " ===");
            System.out.println("Pixels de contour: " + edgePixelCount);
            System.out.println("Centre: (" + centerX + ", " + centerY + ")");
            System.out.println("BGR: B=" + (int)pixelColor[0] + ", G=" + (int)pixelColor[1] + ", R=" + (int)pixelColor[2]);
            System.out.println("HSV: H=" + (int)pixelHSV[0] + ", S=" + (int)pixelHSV[1] + ", V=" + (int)pixelHSV[2]);
        } else {
            System.out.println("\n=== Threshold1=" + threshold1 + ", Threshold2=" + threshold2 + " ===");
            System.out.println("Aucun pixel de contour trouvé!");
            Imgproc.putText(colorSample, "Aucun contour", new Point(50, 75), 
                          Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255), 2);
        }
        
        // Afficher les résultats (images redimensionnées)
        HighGui.imshow("Contours (Canny)", edgesResized);
        HighGui.imshow("Centre des Contours", imageWithCenter);
        HighGui.imshow("Couleur au Centre", colorSample);
    }
    
    public static void main(String[] args) {
        System.out.println("Welcome to OpenCV " + Core.VERSION);
        
        // Charger l'image
        String imagePath = "./ML_Project_Candy/nos_dataset/Entrainement/Croco/PXL_20251015_080424132.RAW-01.COVER.jpg";
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
        namedWindow("Centre des Contours");
        namedWindow("Couleur au Centre");
        
        System.out.println("\n=== Interface Interactive ===");
        System.out.println("Utilisez les touches pour ajuster les seuils (modification +/-10):");
        System.out.println("  '1' / '2' : Diminuer/Augmenter Threshold 1 (actuellement: " + threshold1 + ")");
        System.out.println("  '3' / '4' : Diminuer/Augmenter Threshold 2 (actuellement: " + threshold2 + ")");
        System.out.println("  'r' : Reset (100, 200)");
        System.out.println("  'ESC' : Quitter");
        System.out.println("\nImages redimensionnées à 25% pour un meilleur affichage");
        
        // Affichage initial
        updateEdges();
        
        // Boucle interactive
        while (true) {
            int key = waitKey(0); // Attendre une touche
            
            if (key == 27 || key == -1) { // ESC
                break;
            } else if (key == '1') {
                threshold1 = Math.max(0, threshold1 - 10);
                System.out.println("Threshold 1: " + threshold1);
                updateEdges();
            } else if (key == '2') {
                threshold1 = Math.min(500, threshold1 + 10);
                System.out.println("Threshold 1: " + threshold1);
                updateEdges();
            } else if (key == '3') {
                threshold2 = Math.max(0, threshold2 - 10);
                System.out.println("Threshold 2: " + threshold2);
                updateEdges();
            } else if (key == '4') {
                threshold2 = Math.min(500, threshold2 + 10);
                System.out.println("Threshold 2: " + threshold2);
                updateEdges();
            } else if (key == 'r' || key == 'R') {
                threshold1 = 100;
                threshold2 = 200;
                System.out.println("Reset: Threshold 1=" + threshold1 + ", Threshold 2=" + threshold2);
                updateEdges();
            }
        }
        
        destroyAllWindows();
    }
}