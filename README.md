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

Cette partie va se concentrer sur la partie Unsupervised Learning (Apprentissage non supervisé).

Contrairement à l'apprentissage supervisé, l'apprentissage non supervisé fonctionne avec des données qui ne possèdent pas de labels connus à l'avance. L'IA doit donc essayer de trouver elle-même des ressemblances et des groupes au sein des données.

Dans notre cas, le but sera de faire du **clustering**. Le clustering consiste à regrouper automatiquement des données qui présentent des caractéristiques similaires dans différents groupes appelés **clusters**.

Pour cette partie, nous ne travaillons plus avec les images IRM du dataset précédent. Nous utilisons un dataset contenant des témoignages médicaux, décrivant différents symptômes. Le but est donc de regrouper automatiquement ces témoignages en fonction des symptômes qu'ils contiennent, sans donner à l'algorithme la réponse à l'avance.

### Preprocess

Avant de commencer l'apprentissage, nous devons d'abord préparer notre dataset afin qu'il puisse être compris par les différents algorithmes de Machine Learning.

Notre fichier CSV initial contient des informations qui ne sont pas directement adaptées à notre utilisation. Nous allons donc créer un nouveau DataFrame avec trois colonnes :

- `chiffre` : l'identifiant du témoignage.

- `couleur` : une information présente dans le dataset initial.

- `phrase` : le témoignage contenant les symptômes.

Nous parcourons ensuite le fichier CSV ligne par ligne afin de séparer correctement les différentes informations. Nous supprimons également les guillemets présents dans certaines phrases afin d'obtenir uniquement le texte correspondant aux symptômes.

Une fois cette étape terminée, nous vérifions que les différentes colonnes possèdent bien le même nombre d'éléments et qu'il n'y a pas de valeur vide dans les colonnes importantes.

Notre dataset final contient donc **1011 témoignages**.

## TF-IDF

Les algorithmes de Machine Learning ne peuvent pas directement travailler avec des phrases comme le ferait un humain. Nous devons donc transformer les textes en données numériques.

Pour cela, nous utilisons **TF-IDF** (*Term Frequency-Inverse Document Frequency*).

Le TF-IDF permet de donner un poids aux différents mots en fonction de leur importance dans les témoignages. Un mot qui apparaît souvent dans un témoignage mais qui est relativement rare dans l'ensemble du dataset aura un poids important.

À l'inverse, les mots qui apparaissent dans énormément de témoignages seront considérés comme moins importants.

Nous utilisons également :

```python
TfidfVectorizer(stop_words='english')
```

Les *stop words* sont des mots très courants comme `the`, `a`, `is`, etc. Ils apportent généralement peu d'informations pour différencier les témoignages. Nous les retirons donc afin de conserver principalement les mots importants liés aux symptômes.

Après cette transformation, nos **1011 témoignages sont représentés par 2135 caractéristiques**.

## Truncated 

Après le TF-IDF, nous obtenons un espace vectoriel contenant **2135 dimensions**. Cette quantité de dimensions est importante et peut rendre les calculs plus complexes.

Nous utilisons donc **TruncatedSVD** (*Truncated Singular Value Decomposition*) afin de réduire la dimension de nos données.

Nous testons plusieurs nombres de composantes, de **10 à 100**, et pour chacune de ces configurations nous testons différents nombres de clusters afin de rechercher la combinaison donnant le meilleur résultat avec le **Silhouette Score**.

Après cette comparaison, la meilleure combinaison obtenue est :

- **10 composantes SVD**

- **10 clusters**

- **Silhouette Score : 0.3041**

Nous décidons donc d'utiliser 10 composantes pour la suite de notre apprentissage.

Avec 10 composantes, la variance expliquée est d'environ **8,18 %**. La réduction permet donc de simplifier fortement notre espace de données, mais elle entraîne également une perte d'information. Ce point est important à prendre en compte lors de l'interprétation des résultats.

## Entrainement 

### Elbow Method et Silhouette Score

Pour déterminer le nombre de clusters à utiliser, nous utilisons deux méthodes complémentaires.

Tout d'abord, nous utilisons la **Elbow Method** (méthode du coude).

Elle repose sur l'**inertie**. L'inertie représente la somme des distances au carré entre les données et le centre de leur cluster. KMeans cherche à minimiser cette valeur afin d'obtenir des groupes relativement compacts.

Nous testons différents nombres de clusters et observons l'évolution de l'inertie. Lorsque son amélioration devient moins importante, on peut identifier une zone correspondant au « coude » du graphique.

Nous utilisons ensuite le **Silhouette Score** afin de compléter cette analyse.

Le Silhouette Score permet de mesurer à quel point les données sont bien regroupées dans leurs clusters. Son résultat est compris entre **-1 et 1**. Plus le score est élevé, plus les données d'un même cluster sont proches les unes des autres et plus elles sont éloignées des autres clusters.

Les différentes analyses nous amènent à retenir **10 clusters** pour notre étude.

Ce nombre nous semble également cohérent avec notre objectif : nous cherchons à faire ressortir des familles de symptômes et non à créer un très grand nombre de groupes difficiles à interpréter.

### KMeans

Nous commençons notre comparaison avec **KMeans**.

KMeans est un algorithme de clustering qui cherche à répartir les données dans un nombre défini de groupes. Chaque groupe possède un **centroïde**, qui représente son centre.

Son fonctionnement peut être résumé en plusieurs étapes :

- L'algorithme initialise les centroïdes.

- Chaque témoignage est associé au centroïde dont il est le plus proche.

- Les centroïdes sont ensuite recalculés en fonction des données qui leur sont associées.

- Ces étapes sont répétées jusqu'à ce que les groupes se stabilisent.

Dans notre cas, nous utilisons :

```python
KMeans(n_clusters=10, random_state=42, n_init="auto")
```

Nous choisissons donc **10 clusters** afin de regrouper les 1011 témoignages.

Nous pouvons ensuite observer les mots les plus importants dans chaque cluster. Cela permet d'avoir une première idée de la thématique de chaque groupe.

Par exemple, certains clusters font ressortir des mots associés aux douleurs articulaires, aux problèmes respiratoires, aux symptômes cutanés ou encore aux maux de tête.

Le KMeans obtient finalement un **Silhouette Score de 0.3041**.

### Clustering Hiérarchique Agglomératif

Nous utilisons ensuite le **clustering hiérarchique agglomératif** afin de comparer KMeans avec une méthode fonctionnant différemment.

Contrairement à KMeans, qui commence directement avec un nombre défini de groupes, le clustering agglomératif commence avec chaque témoignage dans son propre cluster.

Dans notre cas, nous commençons donc avec **1011 clusters**.

L'algorithme recherche ensuite les clusters les plus proches et les fusionne progressivement :

1. Chaque témoignage commence comme un cluster individuel.

2. Les clusters les plus proches sont recherchés.

3. Les deux clusters sont fusionnés.

4. Le processus est répété progressivement.

5. L'algorithme s'arrête lorsque nous atteignons les **10 clusters** souhaités.

Nous utilisons également :

```python
linkage='ward'
```

La méthode de Ward cherche à limiter la dispersion des données à l'intérieur des clusters lors des différentes fusions.

Le clustering agglomératif obtient un **Silhouette Score de 0.2821**.

Cette méthode nous permet donc de comparer une approche hiérarchique avec KMeans.

### Gaussian Mixture Model

Nous testons ensuite le **Gaussian Mixture Model (GMM)**.

Contrairement à KMeans, le GMM considère que les données peuvent être représentées par un mélange de plusieurs distributions gaussiennes.

Un des intérêts du GMM est qu'il fonctionne avec une approche probabiliste. Un témoignage peut avoir une probabilité d'appartenance à plusieurs groupes, au lieu d'être considéré comme appartenant de manière totalement certaine à un seul cluster.

Le GMM utilise l'algorithme **Expectation-Maximization (EM)** :

- L'étape **E (Expectation)** estime la probabilité qu'une donnée appartienne à chaque cluster.

- L'étape **M (Maximization)** ajuste ensuite les paramètres des différentes distributions en fonction de ces probabilités.

Ces deux étapes sont répétées jusqu'à obtenir une solution stable.

Nous utilisons :

```python
GaussianMixture(
    n_components=10,
    random_state=42
)
```

Nous choisissons donc 10 composantes afin de conserver le même nombre de groupes que pour les autres modèles.

Le GMM obtient un **Silhouette Score de 0.2350**.

### DBSCAN

Enfin, nous testons **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*).

Contrairement aux modèles précédents, DBSCAN ne cherche pas à répartir toutes les données dans un nombre fixé de clusters.

Il fonctionne en recherchant des zones dans lesquelles les données sont suffisamment concentrées.

Nous utilisons principalement deux paramètres :

- `eps` : définit la distance maximale permettant de considérer deux points comme voisins.

- `min_samples` : définit le nombre minimum de points nécessaires pour considérer une zone comme suffisamment dense.

Un avantage important de DBSCAN est qu'il peut également identifier des données considérées comme du **bruit**. Ces données reçoivent alors le label `-1` et ne sont forcées dans aucun cluster.

Dans notre configuration :

```python
DBSCAN(eps=0.1, min_samples=4)
```

DBSCAN détecte **13 clusters** et considère **391 témoignages comme du bruit**.

Cela représente une partie importante de notre dataset. Son **Silhouette Score est de 0.0245**, ce qui est nettement inférieur aux autres méthodes testées.

## Comparaison des 4 modèles

Afin de comparer les différents algorithmes, nous utilisons le **Silhouette Score**.

Nous obtenons les résultats suivants :

| Modèle        | Silhouette Score | Clusters | Bruit |
| ------------- | ---------------: | -------: | ----: |
| **KMeans**    |       **0.3041** |       10 |     0 |
| Agglomerative |           0.2821 |       10 |     0 |
| GMM           |           0.2350 |       10 |     0 |
| DBSCAN        |           0.0245 |       13 |   391 |

Les résultats montrent des différences entre les quatre méthodes.

KMeans obtient un Silhouette Score de **0.3041**, tandis que le clustering agglomératif obtient **0.2821** et le GMM **0.2350**.

DBSCAN obtient un score beaucoup plus faible de **0.0245**. Il identifie également **391 témoignages comme du bruit**, ce qui montre que les paramètres utilisés ne permettent pas de regrouper efficacement une grande partie des données.

### Conclusion de la partie Unsupervised

Après avoir testé quatre algorithmes de clustering différents, nous pouvons constater que les résultats restent relativement modérés.

Le meilleur Silhouette Score obtenu est de **0.3041 avec KMeans et 10 clusters**.

Cela signifie que les groupes obtenus présentent une certaine structure, mais que les frontières entre les différents clusters restent relativement peu marquées.

Plusieurs raisons peuvent expliquer cela.

- Tout d'abord, plusieurs symptômes peuvent être présents dans différentes pathologies. Un même symptôme peut donc apparaître dans plusieurs groupes de témoignages.

- Ensuite, nous avons fortement réduit la dimension de nos données avec **TruncatedSVD**. Nous sommes passés de **2135 caractéristiques à seulement 10 composantes**, ce qui simplifie les calculs mais entraîne nécessairement une perte d'information.

- Enfin, deux personnes peuvent décrire un même symptôme avec des formulations différentes. Même si le sens est similaire, leurs représentations TF-IDF peuvent être différentes.

Dans notre cas, **KMeans constitue donc notre base de comparaison la plus intéressante selon le Silhouette Score obtenu**, avec 10 clusters et un score de 0.3041.

Cependant, ce résultat ne signifie pas que les clusters correspondent directement à des catégories médicales réelles. Ils représentent avant tout des regroupements statistiques de témoignages présentant des caractéristiques textuelles similaires.

Cette distinction est importante dans le contexte de CVRIE : l'objectif de cette partie non supervisée est d'étudier la structure des données et de voir si des regroupements naturels apparaissent, et non de remplacer un diagnostic médical.

