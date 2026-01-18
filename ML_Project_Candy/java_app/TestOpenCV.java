import org.opencv.core.Core;

public class TestOpenCV {
    public static void main(String[] args) {
        try {
            System.out.println("Tentative de chargement d'OpenCV...");
            System.out.println("java.library.path: " + System.getProperty("java.library.path"));
            System.out.println("Nom de la bibliothèque: " + Core.NATIVE_LIBRARY_NAME);
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            System.out.println("✅ OpenCV chargé avec succès!");
            System.out.println("Version: " + Core.getVersionString());
        } catch (Exception | Error e) {
            System.err.println("❌ Erreur: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
