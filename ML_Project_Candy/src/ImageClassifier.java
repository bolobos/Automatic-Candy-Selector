import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;
import smile.data.type.DataTypes;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.nio.file.Path;
import java.nio.file.Paths;

import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;

public class ImageClassifier {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static double[] imageToVector(String path, int width, int height) {
        Mat img = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
        if (img.empty()) return null;
        Imgproc.resize(img, img, new Size(width, height));
        img.convertTo(img, CvType.CV_64F, 1.0 / 255.0);
        double[] data = new double[(int) (img.total() * img.channels())];
        img.get(0, 0, data);
        return data;
    }

    public static void main(String[] args) {

        // Specify path (relative)
        Path baseDir = Paths.get(args.length > 0 ? args[0] : "notre_dataset");
        System.out.println("Dossier utilisé : " + baseDir);
        File root = new File(baseDir.toString());
        if (!root.exists() || !root.isDirectory()) {
            System.err.println("Le dossier spécifié n'existe pas ou n'est pas un dossier : " + baseDir);
            return;
        }

        int imgWidth = 64;
        int imgHeight = 64;

        List<double[]> featuresList = new ArrayList<>();
        List<Integer> labelsList = new ArrayList<>();
        List<String> labelNames = new ArrayList<>();

        int labelIndex = 0;

        // For to select each folder
        for (File dir : root.listFiles()) {
            if (!dir.isDirectory()) continue;

            // Extract label
            labelNames.add(dir.getName());
            System.out.println("📂 Classe détectée : " + dir.getName());

            // For to select each picture
            for (File file : dir.listFiles()) {
                
                if (file.getName().toLowerCase().endsWith(".jpg")
                        || file.getName().toLowerCase().endsWith(".png")) {
                    
                            // Save picture as vector
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

        // Créer les noms de colonnes pour chaque feature
        String[] colNames = new String[X[0].length];
        for (int i = 0; i < X[0].length; i++) {
            colNames[i] = "f" + i;
        }

        // Créer DataFrame avec les types corrects
        DataFrame df = DataFrame.of(X, colNames);
        df = df.merge(IntVector.of("label", y));

        // Entraînement du modèle Random Forest
        RandomForest model = RandomForest.fit(Formula.lhs("label"), df);
        System.out.println("✅ Modèle entraîné avec succès !");

        // Test sur la première image du DataFrame
        int pred = model.predict(df.get(0));
        System.out.println("🔮 Prédiction exemple (première image) : " + labelNames.get(pred));

        // Test avec une image spécifique
        String testImagePath = "notre_dataset/Schtroumpf/PXL_20251015_080432067.RAW-01.COVER.jpg";
        double[] testVec = imageToVector(testImagePath, imgWidth, imgHeight);
        if (testVec != null) {
            // Créer un DataFrame avec une seule ligne pour la prédiction
            double[][] testData = {testVec};
            DataFrame testDf = DataFrame.of(testData, colNames);
            int predTest = model.predict(testDf.get(0));
            System.out.println("🎯 Prédiction pour " + testImagePath + " : " + labelNames.get(predTest));
        } else {
            System.err.println("❌ Impossible de charger l'image : " + testImagePath);
        }
    }
}
