import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.highgui.HighGui;

class ColorIdentificator3000 {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
    
    public static void main(String[] args) {
        System.out.println("Welcome to OpenCV " + Core.VERSION);
        
        // Charger l'image
        // String imagePath = "./nos_dataset/Entrainement/Croco/PXL_20251015_080355955.RAW-01.COVER.jpg";
        String imagePath = "./nos_dataset/Entrainement/Croco/PXL_20251015_080424132.RAW-01.COVER.jpg";
        Mat image = Imgcodecs.imread(imagePath);
        
        if (image.empty()) {
            System.out.println("Erreur: Impossible de charger l'image: " + imagePath);
            return;
        }
        
        System.out.println("Image chargée avec succès!");
        System.out.println("Dimensions: " + image.cols() + "x" + image.rows());
        
        // Afficher l'image
        HighGui.imshow("Candy Image", image);
        HighGui.resizeWindow("Candy Image", 800, 600);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();
    }
}