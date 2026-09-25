KMeans. C'est notre baseline. Il fonctionne très bien avec nos données après le TruncatedSVD car il cherche à minimiser la distance par rapport à un centre. C'est rapide et efficace pour des clusters de taille à peu près égale.

Clustering Hiérarchique Agglomératif. Au lieu de définir des centres, il construit une hiérarchie de clusters en fusionnant les points les plus proches. C'est super pertinent pour du texte car ça permet de voir les sous-thématiques et on n'est pas obligé de forcer une répartition sphérique comme le fait KMeans.

DBSCAN. C'est un algorithme basé sur la densité. Son gros avantage c'est qu'il n'a pas besoin qu'on lui donne le nombre de clusters à l'avance et il sait identifier le bruit ou les témoignages atypiques qui ne rentrent dans aucune catégorie. C'est un excellent contrepoint aux méthodes qui forcent chaque point dans un groupe.

Modèles de Mélange Gaussien. Contrairement à KMeans qui assigne un point à un seul cluster de façon stricte, le GMM utilise des probabilités. Un symptôme ou un témoignage peut appartenir à plusieurs pathologies avec des degrés de certitude différents. C'est très réaliste pour le domaine médical.

En choisissant ces quatre modèles on couvre les grandes familles du clustering. On a une méthode par centroïde, une hiérarchique, une par densité et une probabiliste. Ça va donner une analyse comparative très riche et justifier parfaitement nos choix dans le notebook.