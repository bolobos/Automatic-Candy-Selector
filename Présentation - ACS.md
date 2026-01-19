Le projet Automatic Candy Selector intervient dans le contexte du cours d'IN451 à l'ESISAR, réalisé par Jean-Baptiste Caignaert.

Le but est le suivant : après avoir donné une image (contenant un bonbon ou pas) à notre programme, celui-ci doit reconnaitre de quel il s'agit. 

> Nous sommes ici dans un cours d'intelligence artificielle donc la détection de ce type de bonbon sera faite avec un algorithme contenant de l'intelligence artificielle. 

---
### Langages utilisés :

- Python : récupération du dataset, création du modèle, réglage de paramètres
- JAVA : interraction avec le modèle, interface utilisateur, annotation du dataset

--- 
### IA utilisée
Deep learning avec YOLO, système fonctionnant avec un **Réseau neuronal convolutif** (CNN).

L'utilisation d'une IA de la sorte est la façon la plus optimale et utilisée pour faire du traitement d'image avec détection et reconnaissance d'objet. Plus robuste qu'un machine learning basé sur du SVM+HOG, la façon d'extraire les données clés de l'images (contours, formes)ce fait tout seul par rapport au machine learning.

Le choix c'est porté sur une IA de la sorte également après différents essais avec d'autres IA (*Voir [Développement - ACS.md](Développement%20-%20ACS.md)*).

---
### Bibliothèques utilisées : 

Bibliothèques Python :
- **Ultralytics YOLO**
- **OpenCV**

Bibliothèques JAVA :
- **OpenCV**

---
## Fonctionnement global

### Construction du modèle
#### Dataset
La première chose va être de créer un dataset. IL faut donc prendre un grand nombres de photos et les placer dans [ML_Project_Candy/python_training/datasets](ML_Project_Candy/python_training/datasets).

Il faut ensuite **annoter** toutes les photos : dire de quelles photos il s'agit. Une première étape est de faire tourner le programme [CandyDetectorApp.java](ML_Project_Candy/java_app/src/CandyDetectorApp.java). Celui-ci nous propose une postion de l'objet et on peut le modifier par l'utilisateur.
#### Entrainement du modèle
Executer le programme [train_yolov8_candy.py](ML_Project_Candy/python_training/scripts/train_yolov8_candy.py). Celui-ci va démarrer l'entrainement du modèle. Il est possible que l'entrainement dure un vingtaine de minutes avec un PC performant.

#### Utilisation du modèle 
Enfin, c'est en exécutant [run.sh](ML_Project_Candy/java_app/run.sh) (qui execute [CandyDetectorApp](ML_Project_Candy/java_app/src/CandyDetectorApp.java)) que l'on va avoir l'interface utilisateur comme si on était à l'entreprise de bonbons.
Via celle-ci, on va pouvoir sélectionner n'importe quelle photo qui va passer dans le modèle généré précédemment.

On va y retrouver le ou les bonbons détéctés avec l'indice de confiance associé. On y retrouve également la couleur avec ces différentes proportions de l'objet et c'est tout !

Vous pouvez retrouver l'image exportée après être passée dans le modèle avec l'objet entourée et labellisé. 

--- 
## Documentation

- [Développement - ACS.md](Développement%20-%20ACS.md)
- [Intelligence artificielle - ACS.md](Intelligence%20artificielle%20-%20ACS.md)
- [README.md](README.md)
- [ML_Project_Candy/java_app/GUIDE_UTILISATION.txt](ML_Project_Candy/java_app/GUIDE_UTILISATION.txt)

