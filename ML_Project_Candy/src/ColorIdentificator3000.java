import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import java.util.ArrayList;
import java.util.List;

class ColorIdentificator3000 {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    
    public static void main(String[] args) {
        System.out.println("Welcome to OpenCV " + Core.VERSION);
        
        // Charger l'image
        // String imagePath = "./nos_dataset/Entrainement/Croco/PXL_20251015_080424132.RAW-01.COVER.jpg";
        String imagePath = "./ML_Project_Candy/nos_dataset/Entrainement/Oeuf/PXL_20251015_080620233.RAW-01.COVER.jpg";
        Mat image = Imgcodecs.imread(imagePath);
        
        if (image.empty()) {
            System.out.println("Erreur: Impossible de charger l'image: " + imagePath);
            return;
        }
        
        System.out.println("Image chargée avec succès!");
        System.out.println("Dimensions: " + image.cols() + "x" + image.rows());
        
        // Afficher l'image
        HighGui.imshow("Image", image);
        HighGui.resizeWindow("Image", 800, 600);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();

        //  Détourage contour
        // Convertir en niveaux de gris
        Mat imgray = new Mat();
        Imgproc.cvtColor(image, imgray, Imgproc.COLOR_BGR2GRAY);
        

        // Détection de contours avec Canny
        System.out.println("\nDétection de contours Canny...");
        Mat edges = new Mat();
        Imgproc.Canny(imgray, edges, 100, 200);
        
        // Afficher l'image originale et les contours Canny côte à côte
        HighGui.imshow("Image (Gris)", imgray);
        HighGui.imshow("Contours (Canny)", edges);
        HighGui.resizeWindow("Image (Gris)", 800, 600);
        HighGui.resizeWindow("Contours (Canny)", 800, 600);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();

        //  Extraction de couleur

    }
}