# gestion d'erreur
import numpy as np

# score = X · poids + biais
# Regulation = éviter que le poid du svm soit trop important
def train_svm(images, labels, learning_rate=0.001, iterations=1000, regularization=0.01):
    nb_images, nb_features = images.shape
    nb_classes = 4

    # Initialiser les poids à zéro et un poids pour chaque pixel et chaque classe
    weights = np.zeros((nb_features, nb_classes))
    bias = np.zeros(nb_classes)

    # Entraîner un SVM pour chaque classe
    for classe in range(nb_classes):

        # La classe actuelle devient 1, les autres deviennent 0
        labels_binaires = np.where(labels == classe, 1, 0)

        # Initialiser les poids du SVM
        poids = np.zeros(nb_features)
        biais = 0

        for i in range(iterations):

            # Calcul du score
            scores = np.dot(images, poids) + biais

            # Vérifier les erreurs, si score trop faible correction avec le modèle
            erreurs = labels_binaires * scores < 1

            # Calcul des corrections
            poids_gradient = regularization * poids
            # .sum additionne et .dot donne un score img x poid = score
            poids_gradient -= np.dot(images[erreurs].T, labels_binaires[erreurs]) / nb_images # Calcul du poid en fonction des errer puis donner une direction
            biais_gradient = -np.sum(labels_binaires[erreurs]) / nb_images # Correction du biais

            # Mise à jour des poids
            poids -= learning_rate * poids_gradient
            biais -= learning_rate * biais_gradient

        # Ranger les valeurs dans une matrice 
        weights[:, classe] = poids
        bias[classe] = biais

    return weights, bias

def predict(images, weights, bias):
    score = np.dot(images, weights) + bias # score de chaque classe

    # Prendre la classe avec le meilleur score
    predictions = np.argmax(score, axis=1)
    return predictions

# Connaissance du modèle
weights, bias = train_svm(X_train_plat, y_train)

# Prédiction du modèle
predictions = predict(X_test_plat, weights, bias)
