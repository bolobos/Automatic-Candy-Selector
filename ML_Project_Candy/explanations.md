**ImageClassifier**

---
Bibliothèques
ia : SMILE
slf4j
libopencv


Questions à ce poser :

- Trop grande représentation de certaines images ?

- Pré-traitement (détourage) puis machine learning ?

Fonctionnement :
- Choix d'un modèle déjà existant ou pas 

- Création d'un nouveau modèle :

- On énumère les dossiers, puis les fichiers dans les dossiers, puis on met dans ces deux éléments : 
            List<double[]> featuresList = new ArrayList<>();
            List<Integer> labelsList = new ArrayList<>();
 les éléments suivants : 
 vec le vecteur d'image
labelIndex l'index de du dossier

- pixel par pixel

Après on cré"é un dataframe contenant les vecteurs et l'index (decrivant le nom de tous les pixels)

Chaque pixel est ici un feature : une caractéristique numérique à prendre en compte


- Utilisation de opencv pour faire la conversion d'image en vecteurs

Les facteurs qui vont vraiment améliorer les résultats, par ordre d'importance :

1. Qualité et quantité des données (80% de l'impact)

Plus d'images d'entraînement = meilleur modèle
Images bien labelisées et variées (différents angles, lumières, tailles)
Aucun label dupliqué ou bruyant
2. Preprocessing intelligent des images

Augmentation de données : rotation, flip, zoom (génère plus de variabilité artificielle)
Détection de contours (Canny, Sobel) : au lieu de pixels bruts, extraire les edges pertinents
Normalisation adaptive (CLAHE) : améliore le contraste localement
Recadrage/centrage : enlever le bruit inutile
3. Meilleure représentation des features

Au lieu de 360k pixels bruts → utiliser HOG, SIFT, ORB (descripteurs qui capturent la structure)
Réduire la dimensionnalité avec PCA (garder seulement 95% de variance)
Ça limite le surapprentissage et accélère
4. Hyperparamètres (10% d'impact)

Node size, trees, features/split : utiles mais secondaires
5. Meilleur algo

CNN (réseau convolutif) : meilleur que RandomForest sur images, mais plus complexe
Conseil : Améliore d'abord tes données (plus + variées + propres) avant de tuner les hyperparamètres.









Pour des images, les modèles simples et optimisés :

1. SVM (Support Vector Machine) + HOG ⭐ Meilleur compromis

Extrait des features HOG (Histogram of Oriented Gradients) des images
SVM trouve l'hyperplan optimal pour séparer les classes
Rapide, peu de surapprentissage, bonne généralisation
Exemple : new SVM<>(X, y) avec features HOG
2. CNN léger (MobileNet, EfficientNet) ⭐⭐ Meilleur mais plus complexe

Apprend directement les features (pas besoin de HOG)
Beaucoup mieux que RandomForest sur images
Rapide si tu utilises un modèle pré-entraîné (transfer learning)
Exemple : importer MobileNet pré-entraîné, juste ré-entraîner la dernière couche
3. Gradient Boosting (XGBoost) + features

Meilleur que RandomForest
Toujours rapide et simple
Bon si tu veux rester en classique ML
4. Transfer Learning + ResNet50

Récupérer un ResNet50 pré-entraîné (ImageNet)
Fine-tuner sur tes bonbons
Meilleur résultat avec peu de code
Mon conseil : commence par SVM + HOG (simple, efficace). Si besoin de mieux → Transfer Learning + MobileNet (meilleur, pas beaucoup plus complexe).

