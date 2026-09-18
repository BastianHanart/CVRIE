from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

# Pourcentage de raison : accuracy = nombre de bonnes prédictions / nombre total de prédictions
# Presicion global : presision, recall (faux negatif), f-1 score (permet de rattraper en cas de trop gros écart)
from sklearn.metrics import accuracy_score, classification_report

# Entrainement / test
from sklearn.model_selection import train_test_split

dataset_path = Path("./archive/Training")
# glioma = 0, meningioma = 1, ...
labels_name = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

def load_data():
    # Image = tableau de pixel, Labels = numéro de classe
    images = []
    labels = []

    # label_id=0, label_name="glioma"
    for label_id, label_name in enumerate(labels_name):
        folder = dataset_path / label_name

        # si dossier existe pas, crash pas et passe à la suite
        if not folder.exists():
            print(f"Dossier manquant : {folder}")
            continue
        
        # Prend tout les .jpg
        for file in folder.glob("*.jpg"):
            # Pillow charge l'image 
            image = Image.open(file)
            # Transforme l'image en tableau de nb entre 0.0 et 1.0
            image = np.array(image, dtype=np.float32) / 255.0
            # ajout à la list d'image 
            images.append(image)
            # ajout à la list des labels
            labels.append(label_id)   
    # converti les list en tabl      
    return np.array(images), np.array(labels)

# entrainement et labels
def train_model(x_train, y_train):
    # Créer et entrainé le modèle
    # pas d'aléatoire et si image en moins grand nombre elles compte plus
    classifier = LogisticRegression(max_iter=1000, random_state=0, class_weight="balanced")
    # entrainement et prédiction
    classifier.fit(x_train, y_train)
    return classifier

# vrai label, vrai prédiction
def display_results(y_true, y_pred):
    score = accuracy_score(y_true, y_pred)
    print("\nRésultat")
    print(f"Précision : {score * 100:.2f}%")
    print("\nRapport détaillé")
    # precision, recall, f-1 score
    print(classification_report(y_true, y_pred, target_names=labels_name))

def main():
    if not dataset_path.exists():
        print("Les données prétraitées sont introuvables.")
        return 84

    print("Chargement des données")
    images, labels = load_data()

    if len(images) == 0:
        print("Aucune donnée disponible")
        return 84

    print(f"{len(images)} images chargées")
    # Transformation des images en vecteurs pour sickit-learn
    features = images.reshape(images.shape[0], -1) 
    # égalité des images 
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0, stratify=labels)

    print(f"Jeu d'entraînement : {len(x_train)} images")
    print(f"Jeu de test : {len(x_test)} images")
    print("\nEntraînement du modèle")
    model = train_model(x_train, y_train)
    predictions = model.predict(x_test)
    display_results(y_test, predictions)
    return 0

if __name__ == "__main__":
    exit(main())
