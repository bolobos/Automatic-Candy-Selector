// Compiler : javac -cp "lib/*" -d bin src/*.java
// Executer : export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH && java -Xmx2048m -cp "bin:lib/*" -Djava.library.path=lib ImageClassifier

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;

import java.io.File;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.List;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Scanner;

import smile.data.vector.IntVector;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;

public class ImageClassifier {
    static {
        System.loadLibrary("opencv_java4100");
    }

    // Convert image (jpeg) to vector with a specify size
    public static double[] imageToVector(String path, int width, int height) {
        Mat img = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
        if (img.empty()) return null;
        Imgproc.resize(img, img, new Size(width, height));
        img.convertTo(img, CvType.CV_64F, 1.0 / 255.0);
        double[] data = new double[(int) (img.total() * img.channels())];
        img.get(0, 0, data);
        return data;
    }

    // Get all names of the candys thanks to the folders names
    public static List<String> extractLabelNames(String datasetPath) {
        List<String> labelNames = new ArrayList<>();
        File root = new File(datasetPath);
        
        if (!root.exists() || !root.isDirectory()) {
            System.err.println("⚠️ Le dossier spécifié n'existe pas : " + datasetPath);
            return labelNames;
        }
        
        File[] dirs = root.listFiles();
        if (dirs != null) {
            for (File dir : dirs) {
                if (dir.isDirectory()) {
                    labelNames.add(dir.getName());
                }
            }
        }
        
        System.out.println("✅ " + labelNames.size() + " classes détectées : " + labelNames);
        return labelNames;
    }


    public static void main(String[] args) {

        // Specify path (relative)
        Path baseDir = Paths.get(args.length > 0 ? args[0] : "nos_dataset/Entrainement");
        System.out.println("Dossier utilisé : " + baseDir);
        
        // Define size : larger size -> larger model
        int imgWidth = 500;
        int imgHeight = 500;
        
        // Extraire les noms de labels depuis les dossiers
        List<String> labelNames = extractLabelNames(baseDir.toString());
        
        RandomForest model = null;
        
        // Vérifier si un modèle existe déjà
        File modelFile = new File("random_forest.model");
        Scanner scanner = new Scanner(System.in);
        
        if (modelFile.exists()) {
            System.out.println("🔍 Un modèle existant a été trouvé.");
            System.out.print("Voulez-vous (1) Charger le modèle existant ou (2) Entraîner un nouveau modèle ? ");
            int choice = scanner.nextInt();
            
            if (choice == 1) {
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream("random_forest.model"))) {
                    model = (RandomForest) ois.readObject();
                    System.out.println("✅ Modèle chargé avec succès !");
                } catch (Exception e) {
                    System.err.println("❌ Erreur lors du chargement : " + e.getMessage());
                    System.out.println("Entraînement d'un nouveau modèle...");
                }
            }
        } else {
            System.out.println("Aucun modèle existant trouvé. Entraînement d'un nouveau modèle...");
        }
        
        // Si pas de modèle chargé, entraîner un nouveau
        if (model == null) {
            File root = new File(baseDir.toString());
            if (!root.exists() || !root.isDirectory()) {
                System.err.println("Le dossier spécifié n'existe pas ou n'est pas un dossier : " + baseDir);
                return;
            }

            List<double[]> featuresList = new ArrayList<>();
            List<Integer> labelsList = new ArrayList<>();
            int labelIndex = 0;

            // For to select each folder
            for (File dir : root.listFiles()) {
                if (!dir.isDirectory()) continue;

                System.out.println("📂 Traitement de la classe : " + dir.getName());

                // For to select each picture
                // Convert all images to vectors
                for (File file : dir.listFiles()) {
                    if (file.getName().toLowerCase().endsWith(".jpg") || file.getName().toLowerCase().endsWith(".png")) {
                        double[] vec = imageToVector(file.getAbsolutePath(), imgWidth, imgHeight);
                        if (vec != null) {
                            featuresList.add(vec);
                            labelsList.add(labelIndex);
                        }
                    }
                }
                labelIndex++;
            }

            double[][] X = featuresList.toArray(new double[0][]);
            int[] y = labelsList.stream().mapToInt(Integer::intValue).toArray();

            System.out.println("📊 Nombre d'images : " + X.length);
            System.out.println("🔢 Taille des vecteurs : " + X[0].length);

            String[] colNames = new String[X[0].length];
            for (int i = 0; i < X[0].length; i++) {
                colNames[i] = "f" + i;
            }

            // df contain all vectors 
            DataFrame df = DataFrame.of(X, colNames);
            df = df.merge(IntVector.of("label", y));

            // Entrainement du modèle
            model = RandomForest.fit(Formula.lhs("label"), df);
            
            System.out.println("✅ Modèle entraîné avec succès !");

            // Sauvegarde du modèle
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("random_forest.model"))) {
                oos.writeObject(model);
                System.out.println("💾 Modèle sauvegardé dans random_forest.model");
            } catch (Exception e) {
                System.err.println("❌ Erreur lors de la sauvegarde : " + e.getMessage());
            }
        }

        // Demander à l'utilisateur de sélectionner une image pour la prédiction
        System.out.println("\n📷 Sélection d'une image pour la prédiction...");
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setCurrentDirectory(new File("nos_dataset/Test"));
        FileNameExtensionFilter filter = new FileNameExtensionFilter(
            "Images (*.jpg, *.jpeg, *.png)", "jpg", "jpeg", "png");
        fileChooser.setFileFilter(filter);
        
        int result = fileChooser.showOpenDialog(null);
        if (result == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            System.out.println("Image sélectionnée : " + selectedFile.getAbsolutePath());
            
            double[] testVec = imageToVector(selectedFile.getAbsolutePath(), imgWidth, imgHeight);
            if (testVec != null && !labelNames.isEmpty()) {
                // Créer un DataFrame pour la prédiction
                String[] colNames = new String[testVec.length];
                for (int i = 0; i < testVec.length; i++) {
                    colNames[i] = "f" + i;
                }
                DataFrame testDf = DataFrame.of(new double[][]{testVec}, colNames);
                int pred = model.predict(testDf.get(0));
                
                // Vérifier que l'index de prédiction est valide
                if (pred >= 0 && pred < labelNames.size()) {
                    System.out.println("🔮 Prédiction pour l'image : " + labelNames.get(pred));
                } else {
                    System.err.println("❌ Index de prédiction invalide : " + pred);
                }
            } else if (testVec == null) {
                System.err.println("❌ Impossible de charger l'image");
            } else {
                System.err.println("❌ Les noms de labels ne sont pas chargés");
            }
        } else {
            System.out.println("❌ Aucune image sélectionnée");
        }
        
        scanner.close();
    }
}
