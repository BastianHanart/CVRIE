# CVRIE - Objectifs du projet

## Le contexte en une phrase

Développer **CVRIE V1**, une intelligence artificielle médicale capable de faire **2 tâches** pour un hôpital : analyser des images médicales (IRM) et regrouper des témoignages de patients par pathologie.

---

## Partie 1 : Apprentissage Supervisé

### L'objectif

Entraîner une IA à **reconnaître le type de tumeur** sur des IRM cérébrales.

### Ce que tu dois faire concrètement

1. **Choisir un dataset d'images médicales**
   - Il doit contenir des images + des labels (ex: "glioma", "meningioma", etc.)
   - Je dois pouvoir justifier mon choix (pourquoi ce dataset ? pourquoi pas un autre ?)
   - Interdit de prendre le même dataset qu'un autre groupe sans justification

2. **Analyser et visualiser le dataset**
   - Afficher des exemples d'images
   - Vérifier la répartition des classes (y a-t-il autant d'images par catégorie ?)
   - Repérer les problèmes : images de tailles différentes, déséquilibre, outliers

3. **Prétraiter les images**
   - Redimensionner toutes les images à la même taille
   - Normaliser les pixels (valeurs entre 0 et 1)
   - Convertir en niveaux de gris ou garder la couleur (justifier ton choix)

4. **Entraîner un modèle de classification**
   - Choisir un algorithme (SVM, Random Forest, etc. avec Scikit-learn)
   - Choisir une **fonction de perte** (loss function) et **l'expliquer** dans le notebook
   - Séparer les données : Training / Validation / Test (de manière stratifiée)

5. **Comparer plusieurs modèles**
   - Tester au moins 2-3 algorithmes différents
   - Documenter les essais, les échecs, les réussites
   - Le jury veut voir la démarche, pas juste le résultat final

6. **Évaluer les performances**
   - Matrice de confusion
   - Courbes ROC
   - Courbes d'apprentissage (loss/accuracy)
   - Visualisation des features (t-SNE, PCA)

### Livrables

- Un notebook `supervised.ipynb` complet
- Un script `.py` qui entraîne le modèle et classifie les données
- Le dataset utilisé (ou un lien vers celui-ci)
- Des graphiques propres et lisibles

---

## Partie 2 : Apprentissage Non-supervisé (Clustering de textes)

### L'objectif

Regrouper automatiquement des **témoignages de patients** par pathologie ou urgence, **sans aucune étiquette**.

### Ce que tu dois faire concrètement

1. **Charger le dataset de témoignages**
   - INTERDIT de modifier le dataset original (pas d'ajout de labels, pas de suppression)
   - Repérer la colonne avec les codes hexadécimaux (`0x000000`) qui servira pour la visualisation finale

2. **Prétraiter le texte (NLP)**
   - Nettoyage : lowercase, suppression ponctuation, stop-words
   - Lemmatisation ou stemming
   - Vectorisation : TF-IDF, CountVectorizer, ou embeddings

3. **Appliquer des algorithmes de clustering**
   - **K-Means** : trouver le nombre optimal de clusters (Elbow method, Silhouette)
   - Tester d'autres algorithmes (Agglomerative, GMM) pour comparer

4. **Évaluer sans labels**
   - Métriques internes : Silhouette Score, Davies-Bouldin
   - Analyse sémantique : lister les mots-clés de chaque cluster pour leur donner un sens médical
   - Tu peux étiqueter manuellement un petit échantillon pour vérifier, mais ne PAS l'injecter dans le modèle

5. **Visualiser les résultats**
   - Réduction de dimension (PCA, t-SNE, UMAP) pour projeter en 2D
   - **Graphique final avec les vraies couleurs hexadécimales** du dataset
   - Le jury révélera la "vraie couleur" lors de la soutenance pour vérifier si tes clusters sont cohérents

### Livrables

- Un notebook `unsupervised.ipynb` complet
- Un script `.py` qui output le clustering
- Le fameux graphique coloré avec les codes hexadécimaux

---

## Contraintes techniques OBLIGATOIRES

### Organisation des fichiers
cvrie/
├── README.md <- Documentation du projet
├── goal.md <- Ce fichier
├── preprocess.py <- Script de prétraitement
├── supervised.py <- Script partie 1
├── unsupervised.py <- Script partie 2
├── notebook_supervised.ipynb <- Notebook partie 1
├── notebook_unsupervised.ipynb <- Notebook partie 2
├── bonus/ <- Dossier bonus (même vide au début)
└── archive/ <- Ton dataset (à ne PAS commiter)

### Ce qui est INTERDIT

- Fichiers binaires, temporaires, `.o`, `.pyc`, etc.
- Fichiers `Zone.Identifier` (Windows)
- Dossier `.venv` ou `__pycache__`
- Dataset trop volumineux

### Ce qui est OBLIGATOIRE

- **Exit code 84** en cas d'erreur, **0** si succès
- Messages d'erreur sur **stderr**
- Scripts `.py` propres et fonctionnels
- Notebooks bien structurés avec sections claires

---

## Structure attendue des notebooks

Chaque notebook doit contenir ces sections clairement identifiées :

### Notebook Supervised

1. **Introduction** : présentation du dataset et justification du choix
2. **Analyse du dataset** : visualisations, statistiques, problèmes détectés
3. **Prétraitement** : redimensionnement, normalisation, gestion du déséquilibre
4. **Fonction de perte** : explication mathématique ET intuitive
5. **Entraînement** : split des données, premiers modèles
6. **Comparaison des modèles** : plusieurs algorithmes testés, résultats
7. **Évaluation** : métriques, graphiques (ROC, confusion matrix, etc.)
8. **Conclusions et insights** : ce que tu as appris, limites du modèle

### Notebook Unsupervised

1. **Introduction** : présentation du dataset de témoignages
2. **Prétraitement NLP** : nettoyage, vectorisation
3. **Clustering** : K-Means + algorithmes alternatifs
4. **Évaluation** : métriques internes, analyse sémantique
5. **Visualisation** : PCA/t-SNE + graphique avec couleurs hexadécimales
6. **Conclusions et insights** : cohérence des clusters, interprétation médicale

---

## Ce qui sera évalué

### Points vérifiés (checklist du sujet)

- [ ] Notebook pour chaque partie
- [ ] Code Python qui fonctionne (supervised + unsupervised)
- [ ] Dataset utilisé pour la partie supervised
- [ ] Section "Prétraitement" claire dans chaque notebook
- [ ] Section "Comparaison des modèles" claire dans chaque notebook
- [ ] Section "Conclusions et insights" claire dans chaque notebook
- [ ] Explication claire de la loss function
- [ ] Beaux graphiques (titres, légendes, lisibles)

### Ce qui fait la différence

- **La qualité de ta soutenance** (plus importante que les résultats bruts)
- **Ta capacité à justifier tes choix** (pourquoi cet algo ? pourquoi cette loss ?)
- **Ta démarche scientifique** (essais, erreurs, itérations documentées)
- **La propreté du code** (exit 84, gestion d'erreurs, pas de fichiers inutiles)
- **La qualité des visualisations** (graphiques professionnels, pas basiques)

---

## Planning à respecter

### Semaine 1 (Mardi 15 au Vendredi 18 Septembre)
- Partie Supervised terminée

### Semaine 2 (Lundi 21 au Vendredi 25 Septembre)
- Partie Unsupervised terminée
- Soutenance préparée

---

## Conseils pour réussir

1. **Documente TOUT** : le jury veut voir ta réflexion, pas juste le code final
2. **Justifie tes choix** : pourquoi cet algo ? pourquoi ces paramètres ?
3. **Soigne tes graphiques** : titres, légendes, couleurs cohérentes
4. **Teste plusieurs approches** : le trial & error documenté est valorisé
5. **Prépare ta soutenance** : c'est là que tu gagnes (ou perds) des points
6. **Respecte les contraintes techniques** : exit 84, pas de fichiers inutiles, etc.

---

## Pièges à éviter

- Ne pas gérer les erreurs (exit 84)
- Commiter des fichiers inutiles (.venv, __pycache__, Zone.Identifier)
- Modifier le dataset de la partie unsupervised
- Utiliser la colonne "0x000000" pour entraîner le modèle
- Notebook sans explications (juste du code sans contexte)
- Graphiques moches ou illisibles
- Ne pas expliquer la loss function

---
