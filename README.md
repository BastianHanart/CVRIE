# CVRIE

## Contexte

Le but du projet CVRIE est de créer une intelligence artificielle conçue pour aider au diagnostic médical en analysant des IRM cérébrales. Pour que l'IA puisse apprendre efficacement, elle a besoin de recevoir des images propres, uniformes et bien formatées.

C'est un projet de Machine Learning. C'est un sous-domaine de l'IA. Au lieu de programmer des règles fixes, on entraîne un algorithme avec des données pour qu'il apprenne à prédire ou décider par lui-même.

Pour réaliser ce projet, nous avons deux types d'apprentissages avec lequel nous renforçons notre IA. Tout d'abord, nous avons un apprentisage supervisé. C'est un apprentissage où l'on utilise des données étiquetées (avec des réponses connues). Et ensuite, un apprentissage non supervisé est au contraire là, nos données n'était pas étiquetées.

## Supervised Learning

Cette partie va se concentrer sur la partie Supervised Learning (Apprentissage supervisé). 
Le but de notre algorithme supervisé sera de faire une classification. Une classification est une méthode d'apprentissage automatique supervisé qui consiste à attribuer une étiquette ou une catégorie à une donnée d'entrée en fonction de caractéristiques apprises

Dans notre cas, c'est plus précisément une classification multiclasse. C'est à dire que le modèle choisit une seule étiquette parmi plus de deux options possiblles

### Choix du Dataset

Pour cette partie de l'entrainement, il nous fallait trouver un dataset d'images médicales. Nous avons donc choisi un dataset d'IRM cérébral. Ce dataset contient 4 types de tumeurs cérébrale. Des gliomes, des méningites, des pituitaires (ou tumeurs hypophysaires) et enfin des images de cerveaux sains (sans tumeurs).

Nous avons choisi ce dataset car il traitait un sujet important pour nous : 

- Encore aujourd'hui, les tumeurs cérébrales sont l'une des causes de mortalité les plus importantes dans le monde, c'est d'ailleurs la première cause de mortalité par cancer chez les enfants et les jeunes adultes.
On comptabilise environ 321 400 nouveaux cas, et près de 248 300 décès par an, dont 6 000 nouveaux cas et 4000 décès en France !

### Parcours & Preprocess

Avant de commencer à utiliser le dataset, nous avons décidé de le parcourir pour pouvoir connaitre toutes ses spécifications. Nous avons donc commencer par afficher les informations relatifs au dataset directement via : 

```python
df.head()
df.info()
df.describe()      
```

Qui nous permettent d'obtenir sa classe, ses infos sur son nombre d'entrée et sa description comme son nombre de colonnes, ses types d'infos à l'interieur ...

Ensuite, on s'interesse aux images plus précisément. On va parcourir tous les fichiers pour regarder le nombre d'images totale et le nombre d'images de chaques tumeurs. On voit donc qu'il y a 5600 images réparti en 1400 images par tumeurs, pour notre problème c'est otpimale, car on ne va pas avoir des données déséquilibrées.

En sachant tout ça on se concentre plus précisément sur les caractéristiques des images. On va récupérer des informations importantes pour l'entrainement, comme les dimensions, la colorimétrie et les différents formats de compression des images. Pour entrainer le plus efficacement l'IA, nous allons chercher à normaliser les images.

Après d'avoir observer tout ça, nous décidons plusieurs choses:

- D'abord de passé toutes les images en nuances de gris. Le RGB sur des IRM est limités et n'apporte aucune information importante dans la recherche   de la tumeurs. Ce qui signifie qu'en passant tout en nuances de gris, on gagne en efficacité sans perdre en qualité. 

Nous allons également normaliser les données de couleurs de 0 à 1 (au lieu de 0 à 255)
Pourquoi ? Car cela permet une accélération de la convergence : Les algorithmes d'optimisation comme la descente de gradient convergent beaucoup plus rapidement lorsque les fonctionnalités ont des échelles similaires. <br>
Mais aussi une équité entre les variables : Les algorithmes basés sur les distances (comme les k-plus proches voisins ou les K-means) ne seront plus biaisés par les variables ayant de grandes valeurs numériques.

- Ensuite, d'avoir deux dimensions pour faire deux entrainements et voir les différences. Un entrainement sur des images en (512,512) et un entrainement sur des images en (128,128). On verra que les deux on des avantages et des inconvénients.

Normaliser les dimensions est indispensable pour permettre aux réseaux de neurones de traiter les lots d'images de manière uniforme et efficace

Cepandant les proportions ne sont pas toujours les mêmes, nous obligeant à ajouter des paddings pour éviter de tordre l'image ou de la coupé. Les paddings ajoutés sont en noir, tout simplement car le seul éléments à être noir sur un IRM cérébral est tout ce qui ne fait pas partie du cerveau et donc ce qui n'est pas important.

- Et enfin, pour terminer ce preprocess, on va venir faire un ".flatten()", qui va venir transformer les dimensions des images en un vecteurs mathématiques. En effet, une IA ne peut travailler qu'avec des vecteurs mathématique, pour elle, des dimensions en 2D d'une images ne veut rien dire.


### Entrainement

Pour entrainer l'IA, nous avons décider de faire 4 modèles différents : 

- Un KNN (K-nearests neighbords (K plus proche voisins)). Ce modèle sert à classer des données ou à prédire une valeur numérique. Et c'est justement ce que l'on veut ici, on veut que notre IA soit capable de classer des images en différentes catégorie de tumeurs. 

- Un Random Forest. Ici, il sert à faire des prédictions précises et robustes en combinant plusieurs arbres de décision. Un peu à l'instar du KNN, ce modèle essaye de prédire les catégorie de tumeurs à laquelle les images appartiennent.

La différence entre les deux ? Le KNN, c'est un apprentissage paresseux. Il ne construit pas de modèle lors de l'entraînement. Pour prédire une valeur, il calcule la distance entre le nouveau point et tous les points d'entraînement. Alors que le Random Forest est une méthode d'ensemble. Il crée une multitude d'arbres de décision pendant l'entraînement en introduisant de l'aléa. La prédiction finale est obtenue par vote majoritaire.

- Un SVM

- Un Logistic Regression

Pour chaque modèle, on fait une matrice de confusion qui nous permet de voir les résultats de ceux-ci. On y voit le nombre de fois ou le modèle à répondu correctement et les moments où il a répondu un mauvais type de tumeurs.

Aussi, pour ce qui est de la séparation des données pour l'entrainement et les test, nous faisons un 80/20.

Pourquoi ? Notre quantité de valeurs présente est de 5600 images, pour avoir assez d'élémént d'entrainement tout en gardant une bonne quantité de test, 80/20 et donc 4480 images d'entrainement et 1120 images de tests restent un des meilleurs ratio pour notre problème. 

### Comparaison et Conclusion

Enfin, nous comparons tous les modèles afin de voir lequel est le plus efficace dans notre cas. Nous comparons leurs résultats et voyons danns quels cas ils ont eu de meilleurs prédictions et où ils ont pu se tromper. 

Ce qui nous permet ensuite de conclure sur le choix à faire pour prédire au mieux la présence de tumeurs dans le cerveau des patients.

Dans notre cas, le modèle Random Forest semble légèrement plus efficace avec un résultat de 0.91152 sur les images de dimensions (512,512)




## Unsupervised Learning 

Et cette partie va se concentrer sur la deuxième étape, le Unsupervised Learning (Apprentissage non supervisé). 
Le but de notre algorithme sera de regrouper les patients en différents groupes de symptômes afin de les prendre en charges plus rapidement et efficacement. 

### Dataset

Pour cette partie de l'entrainement, Nous avions à disposition un dataset non labelisé. Il contenait trois 3 types d'informations :

- Le nombre de la demande

- Une couleur

- Et le symptôme du patient

### Preprocess

Encore une fois, avant de commencer à utiliser le dataset, nous avons décidé de le parcourir pour pouvoir appercevoir des problèmes directement visible.

Nous avons remarquer 2 choses : 

- D'abord la présence de guillemets sur certains des symptômes

- Et l'information de couleur qui n'est pas toujours présente

Nous avons donc décider d'enlever les guillemts qui ne servait en rien à la bonne utilisation des données. Mais de garder les couleurs qui peuvent potentiellements servir par la suite.

### Entrainement

Pour entrainer l'IA, nous avons décider de faire 3 modèles différents : 

- Un K-means, c'est un algorithme d'apprentissage automatique non supervisé qui permet de regrouper des données non labelisés en k groupes distincts appelés clusters en fonction de leur similarité.

- 


- 

Pour vérifier les différents résultat, ou s'assurer de la véracité des test, nosu effectuons des elbow method et des silhouette score qui permette de montrer le lien entre nombre de clusters et efficacité du projet.

### Comparaison et Conclusion



