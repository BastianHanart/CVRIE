# CVRIE - Prétraitement des images médicales

## Contexte

CVRIE est une intelligence artificielle conçue pour aider au diagnostic médical en analysant des IRM cérébrales. Pour que l'IA puisse apprendre efficacement, elle a besoin de recevoir des images **propres, uniformes et bien formatées**.

C'est exactement ce que fait ce script : il prend nos 7200 images IRM brutes et les prépare pour qu'elles soient toutes identiques en taille, en couleur et en intensité.

---

## Pourquoi prétraiter les images ?

Imagine que tu doives comparer 7200 photos de chats, mais que :
- Certaines sont de taille plus grande, d'autre plus petite
- Certaines sont en noir et blanc, d'autres en couleur
- Certaines sont trop sombres, d'autres trop claires

On ne peux donc pas les comparer correctement. C'est exactement le même problème pour l'IA. Le prétraitement permet de mettre toutes les images au même format pour que l'IA puisse se concentrer sur le fais de détecter les erreurs.

---

## Ce que fait le script, étape par étape

### Conversion en niveaux de gris

**Avant :** Les images IRM peuvent avoir différentes palettes de couleurs selon leur provenance.  
**Après :** Toutes les images sont en noir et blanc.

**Pourquoi ?** Les IRM médicales n'ont pas besoin de couleur pour être analysées. L'information importante se trouve dans les **contrastes**. Convertir en niveaux de gris simplifie le travail de l'IA et réduit la quantité de données à traiter, sans perdre d'information utile.

---

### Redimensionnement uniforme

**Avant :** Les images ont des tailles variées.  
**Après :** Toutes les images font exactement **128x128 pixels**.

**Pourquoi ?** Les algorithmes d'intelligence artificielle ont besoin de recevoir des données **toujours identiques en taille**. Si on leur donne des images de tailles différentes, ils ne savent pas les comparer. On choisit 128x128 car c'est un bon compromis : assez grand pour voir les détails des tumeurs, assez petit pour que le traitement reste rapide.

---

### Normalisation des valeurs

**Avant :** Chaque pixel a une valeur entre **0 et 255**.  
**Après :** Chaque pixel a une valeur entre **0.0 et 1.0**.

**Pourquoi ?** Les algorithmes d'IA fonctionnent beaucoup mieux avec des petits nombres. Travailler avec des valeurs entre 0 et 1 permet :
- D'**accélérer** l'apprentissage de l'IA
- D'**éviter les erreurs** de calcul dues aux grands nombres
- De rendre les modèles plus **stables** et **fiables**

C'est comme passer d'une échelle de 0 à 255 à une échelle de 0% à 100% : c'est plus simple à comprendre pour tout le monde, y compris pour l'IA !

---

## Organisation des données

Le script préserve la structure originale du dataset :

archive/<-  Données brutes (avant)
├── Training/
│ ├── glioma/ (tumeurs gliales)
│ ├── meningioma/ (méningiomes)
│ ├── notumor/ (pas de tumeur)
│ └── pituitary/ (tumeurs pituitaires)
└── Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

processed/<-  Données traitées (après)
├── Training/
│ ├── glioma/-> fichiers .npy
│ ├── meningioma/-> fichiers .npy
│ ├── notumor/-> fichiers .npy
│ └── pituitary/-> fichiers .npy
└── Testing/
├── glioma/-> fichiers .npy
├── meningioma/-> fichiers .npy
├── notumor/-> fichiers .npy
└── pituitary/-> fichiers .npy


**Pourquoi sauvegarder en `.npy` ?**  
Le format `.npy` est un format spécial pour les données numériques. Il permet de charger les images **beaucoup plus rapidement** que le format JPG classique, ce qui fait gagner un temps précieux lors de l'entraînement de l'IA.

---

## Comment utiliser le script

### Installation des dépendances

```bash
python3 preprocess.py
```

## Résultat attendu

```
=== Training ===
[Training/glioma] 100 images...
[Training/glioma] 200 images...
...
-> 2870 processed, 0 errors

=== Testing ===
[Testing/glioma] 100 images...
...
-> 764 processed, 0 errors

=== TOTAL: 3634 processed, 0 errors ===
```
