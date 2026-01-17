// Compiler : javac -cp "lib/*" -d bin src/*.java
// Executer : export LD_LIBRARY_PATH=$PWD/lib/opencv:$LD_LIBRARY_PATH && java -Xmx8000m -cp "bin:lib/*" -Djava.library.path=lib/opencv ImageClassifier

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.HOGDescriptor;
import smile.classification.RandomForest;
import smile.classification.SVM;
import smile.classification.OneVersusOne;
import smile.math.kernel.GaussianKernel;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.base.cart.SplitRule;

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

    public static double[] imageToVector(String path, int width, int height) {
        Mat img = Imgcodecs.imread(path, Imgcodecs.IMREAD_GRAYSCALE);
        if (img.empty()) return null;
        Imgproc.resize(img, img, new Size(width, height));
        img.convertTo(img, CvType.CV_64F, 1.0 / 255.0);
        double[] data = new double[(int) (img.total() * img.channels())];
        img.get(0, 0, data);
        return data;
    }

    // Extraire les features HOG + histogramme de couleur (meilleur pour bonbons colorés)
    public static double[] imageToHOG(String path, int width, int height) {
        // Charger l'image en COULEUR pour l'histogramme
        Mat imgColor = Imgcodecs.imread(path);
        if (imgColor.empty()) return null;
        
        // Convertir en grayscale pour HOG
        Mat imgGray = new Mat();
        Imgproc.cvtColor(imgColor, imgGray, Imgproc.COLOR_BGR2GRAY);
        
        // Redimensionner à une taille plus petite pour HOG
        int hogSize = 128;
        Imgproc.resize(imgGray, imgGray, new Size(hogSize, hogSize));
        Imgproc.resize(imgColor, imgColor, new Size(hogSize, hogSize));
        
        // ===== PARTIE 1: HOG (contours/forme) =====
        Size winSize = new Size(hogSize, hogSize);
        Size blockSize = new Size(16, 16);
        Size blockStride = new Size(16, 16);
        Size cellSize = new Size(8, 8);
        int nbins = 9;
        
        HOGDescriptor hog = new HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);
        MatOfFloat hogDescriptors = new MatOfFloat();
        hog.compute(imgGray, hogDescriptors);
        
        float[] hogArray = hogDescriptors.toArray();
        
        // ===== PARTIE 2: Histogramme de couleur (HSV) =====
        Mat imgHSV = new Mat();
        Imgproc.cvtColor(imgColor, imgHSV, Imgproc.COLOR_BGR2HSV);
        
        // Calculer histogramme pour chaque canal HSV
        int histSize = 32;  // 32 bins par canal
        MatOfInt histSizeM = new MatOfInt(histSize);
        MatOfFloat ranges = new MatOfFloat(0f, 180f);  // H: 0-180
        MatOfFloat rangesSV = new MatOfFloat(0f, 256f);  // S,V: 0-256
        
        java.util.List<Mat> hsvChannels = new java.util.ArrayList<>();
        Core.split(imgHSV, hsvChannels);
        
        Mat histH = new Mat();
        Mat histS = new Mat();
        Mat histV = new Mat();
        
        Imgproc.calcHist(java.util.Arrays.asList(hsvChannels.get(0)), new MatOfInt(0), new Mat(), histH, histSizeM, ranges);
        Imgproc.calcHist(java.util.Arrays.asList(hsvChannels.get(1)), new MatOfInt(0), new Mat(), histS, histSizeM, rangesSV);
        Imgproc.calcHist(java.util.Arrays.asList(hsvChannels.get(2)), new MatOfInt(0), new Mat(), histV, histSizeM, rangesSV);
        
        // Normaliser les histogrammes
        Core.normalize(histH, histH, 0, 1, Core.NORM_MINMAX);
        Core.normalize(histS, histS, 0, 1, Core.NORM_MINMAX);
        Core.normalize(histV, histV, 0, 1, Core.NORM_MINMAX);
        
        // ===== PARTIE 3: Combiner HOG + Color =====
        double[] result = new double[hogArray.length + 3 * histSize];
        
        // Copier HOG features
        for (int i = 0; i < hogArray.length; i++) {
            result[i] = hogArray[i];
        }
        
        // Copier histogrammes couleur
        int offset = hogArray.length;
        for (int i = 0; i < histSize; i++) {
            result[offset + i] = histH.get(i, 0)[0];
            result[offset + histSize + i] = histS.get(i, 0)[0];
            result[offset + 2 * histSize + i] = histV.get(i, 0)[0];
        }
        
        // Normaliser L2 l'ensemble des features
        double norm = 0.0;
        for (double v : result) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        
        if (norm > 1e-10) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= norm;
            }
        }
        
        return result;
    }

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
        
        int imgWidth = 600;
        int imgHeight = 600;
        
        // Extraire les noms de labels depuis les dossiers
        List<String> labelNames = extractLabelNames(baseDir.toString());
        
        Scanner scanner = new Scanner(System.in);
        
        // Choix du type de modèle
        System.out.println("\n🤖 Quel type de modèle voulez-vous utiliser ?");
        System.out.println("  (1) Random Forest");
        System.out.println("  (2) SVM (Support Vector Machine)");
        System.out.print("Votre choix : ");
        int modelType = scanner.nextInt();
        
        // Choix du type de features
        System.out.println("\n🎨 Quel type de features voulez-vous utiliser ?");
        System.out.println("  (1) Pixels bruts (360k features)");
        System.out.println("  (2) HOG + Couleur HSV (~2.4k features HOG + 96 color, recommandé pour bonbons)");
        System.out.print("Votre choix : ");
        int featureType = scanner.nextInt();
        
        String modelFileName = (modelType == 1) ? "random_forest" : "svm";
        modelFileName += (featureType == 1) ? "_pixels.model" : "_hog.model";
        Object model = null;
        
        // Variables pour stocker les données d'entraînement (pour test de précision)
        double[][] trainingX = null;
        int[] trainingY = null;
        
        // Vérifier si un modèle existe déjà
        File modelFile = new File(modelFileName);
        
        if (modelFile.exists()) {
            System.out.println("🔍 Un modèle " + (modelType == 1 ? "Random Forest" : "SVM") + " existant a été trouvé.");
            System.out.print("Voulez-vous (1) Charger le modèle existant ou (2) Entraîner un nouveau modèle ? ");
            int choice = scanner.nextInt();
            
            if (choice == 1) {
                try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFileName))) {
                    model = ois.readObject();
                    System.out.println("✅ Modèle chargé avec succès !");
                } catch (Exception e) {
                    System.err.println("❌ Erreur lors du chargement : " + e.getMessage());
                    System.out.println("Entraînement d'un nouveau modèle...");
                }
            }
        } else {
            System.out.println("Aucun modèle " + (modelType == 1 ? "Random Forest" : "SVM") + " existant trouvé. Entraînement d'un nouveau modèle...");
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

                int imagesInFolder = 0;
                // For to select each picture
                for (File file : dir.listFiles()) {
                    if (file.getName().toLowerCase().endsWith(".jpg")
                            || file.getName().toLowerCase().endsWith(".png")) {
                        // Extraction des features selon le type choisi
                        double[] vec = (featureType == 1) 
                            ? imageToVector(file.getAbsolutePath(), imgWidth, imgHeight)
                            : imageToHOG(file.getAbsolutePath(), imgWidth, imgHeight);
                        
                        if (vec != null) {
                            featuresList.add(vec);
                            labelsList.add(labelIndex);
                            imagesInFolder++;
                        }
                    }
                }
                // N'incrémenter labelIndex que si au moins une image a été trouvée
                if (imagesInFolder > 0) {
                    labelIndex++;
                }
            }

            // Convertion liste -> double[] ou int[]
            double[][] X = featuresList.toArray(new double[0][]);
            int[] y = labelsList.stream().mapToInt(Integer::intValue).toArray();
            
            // Sauvegarder pour le test de précision
            trainingX = X;
            trainingY = y;

            System.out.println("📊 Nombre d'images : " + X.length);
            System.out.println("🔢 Taille des vecteurs : " + X[0].length);
            System.out.println("🎨 Type de features : " + (featureType == 1 ? "Pixels bruts" : "HOG+Couleur HSV"));
            
            // Debug : afficher les labels uniques
            java.util.Set<Integer> uniqueLabels = new java.util.HashSet<>();
            for (int label : y) uniqueLabels.add(label);
            System.out.println("🏷️  Labels présents : " + uniqueLabels);


            String[] colNames = new String[X[0].length];
            for (int i = 0; i < X[0].length; i++) {
                colNames[i] = "f" + i;
            }

            // Tableau avec toutes les images décomposées en pixels 
            DataFrame df = DataFrame.of(X, colNames);
            // Rajout de l'index pour chaque image
            df = df.merge(IntVector.of("label", y));

            // lhs précise la colonne qui sera la prédiction
            // Ici c'est la classification qui est utilisé , et non pas une régression (on veux ici choisir entre plusieurs "classes")
            
            if (modelType == 1) {
                // Random Forest
                model = RandomForest.fit(
                    Formula.lhs("label"), 
                    df,
                    500,                // ntrees: 500 arbres
                    600,                // mtry: √360000 ≈ 600 features à tester par split
                    SplitRule.GINI,     // critère de split (GINI ou ENTROPY)
                    20,                 // maxDepth: profondeur max des arbres
                    500,                // maxNodes: nombre max de noeuds par arbre
                    5,                  // nodeSize: min 5 images par feuille
                    1.0                 // subsample: proportion de données (1.0 = toutes)
                );
                System.out.println("✅ Modèle Random Forest entraîné avec succès !");
            } else {
                // SVM avec kernel Gaussien (RBF) en stratégie One-vs-One
                int numClasses = 0;
                for (int label : y) {
                    if (label > numClasses) numClasses = label;
                }
                numClasses++; // +1 car labels commencent à 0
                
                System.out.println("🔢 Nombre de classes détectées pour SVM : " + numClasses);
                System.out.println("⏳ Entraînement SVM multiclasse (One-vs-One)...");
                
                // Gamma réduit pour éviter overfitting sur classe majoritaire
                // Pour 2304 features : 1/n_features = 4.34e-4, on utilise 10x moins
                double gamma = 0.1 / X[0].length;  // Kernel moins flexible
                System.out.println("🎯 Gamma du kernel RBF : " + gamma);
                
                // Créer un classifieur SVM avec C plus élevé pour pénaliser les erreurs
                model = OneVersusOne.fit(X, y, (x, y_binary) -> 
                    SVM.fit(x, y_binary, new GaussianKernel(gamma), 10.0, 0.001)
                );
                System.out.println("✅ Modèle SVM entraîné avec succès !");
                System.out.println("⚙️ Paramètres : gamma=" + String.format("%.6e", gamma) + ", C=10.0");
                
                // Afficher distribution réelle des classes
                int[] classCounts = new int[numClasses];
                for (int label : y) {
                    classCounts[label]++;
                }
                System.out.print("📊 Distribution : ");
                for (int i = 0; i < numClasses; i++) {
                    System.out.print(labelNames.get(i) + "=" + classCounts[i]);
                    if (i < numClasses - 1) System.out.print(", ");
                }
                System.out.println();
            }

            // Sauvegarde du modèle
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFileName))) {
                oos.writeObject(model);
                System.out.println("💾 Modèle sauvegardé dans " + modelFileName);
            } catch (Exception e) {
                System.err.println("❌ Erreur lors de la sauvegarde : " + e.getMessage());
            }
        }

        // ===== TEST DE PRÉCISION SUR DONNÉES D'ENTRAÎNEMENT =====
        if (trainingX != null && trainingY != null) {
            System.out.println("\n📊 Test de précision sur les données d'entraînement...");
            int correctPredictions = 0;
            int totalPredictions = 0;
            
            // Tester sur un échantillon des données d'entraînement
            for (int i = 0; i < Math.min(trainingY.length, 50); i++) {
                int pred;
                if (modelType == 1) {
                    String[] colNames = new String[trainingX[i].length];
                    for (int j = 0; j < trainingX[i].length; j++) {
                        colNames[j] = "f" + j;
                    }
                    DataFrame testDf = DataFrame.of(new double[][]{trainingX[i]}, colNames);
                    pred = ((RandomForest) model).predict(testDf.get(0));
                } else {
                    @SuppressWarnings("unchecked")
                    OneVersusOne<double[]> svmModel = (OneVersusOne<double[]>) model;
                    pred = svmModel.predict(trainingX[i]);
                }
                
                if (pred == trainingY[i]) {
                    correctPredictions++;
                }
                totalPredictions++;
            }
            
            double accuracy = (double) correctPredictions / totalPredictions * 100;
            System.out.println("🎯 Précision sur entraînement (échantillon) : " + String.format("%.1f%%", accuracy) + 
                              " (" + correctPredictions + "/" + totalPredictions + ")");
            
            if (accuracy < 50) {
                System.out.println("⚠️  ATTENTION : Précision très faible ! Le modèle n'a pas bien appris.");
                System.out.println("💡 Recommandations :");
                System.out.println("   - Augmenter C (actuellement 10.0, essayer 100.0)");
                System.out.println("   - Vérifier la qualité des images d'entraînement");
                System.out.println("   - Considérer Random Forest à la place de SVM");
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
            
            // Utiliser le même type de features que pour l'entraînement
            double[] testVec = (featureType == 1)
                ? imageToVector(selectedFile.getAbsolutePath(), imgWidth, imgHeight)
                : imageToHOG(selectedFile.getAbsolutePath(), imgWidth, imgHeight);
            
            if (testVec != null && !labelNames.isEmpty()) {
                // Créer un DataFrame pour la prédiction
                String[] colNames = new String[testVec.length];
                for (int i = 0; i < testVec.length; i++) {
                    colNames[i] = "f" + i;
                }
                int pred;
                if (modelType == 1) {
                    // Random Forest utilise DataFrame
                    DataFrame testDf = DataFrame.of(new double[][]{testVec}, colNames);
                    pred = ((RandomForest) model).predict(testDf.get(0));
                } else {
                    // SVM utilise directement le vecteur
                    @SuppressWarnings("unchecked")
                    OneVersusOne<double[]> svmModel = (OneVersusOne<double[]>) model;
                    pred = svmModel.predict(testVec);
                }
                
                // Vérifier que l'index de prédiction est valide
                if (pred >= 0 && pred < labelNames.size()) {
                    System.out.println("🔮 Prédiction : " + labelNames.get(pred) + " (label " + pred + ")");
                    System.out.println("📊 Nombre de features utilisées : " + testVec.length);
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
