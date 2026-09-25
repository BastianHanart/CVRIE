# gestion d'erreur 
# gerer pour 4 classes et non 2
import numpy as np

# compression de valeur entre 0 et 1
def sigmoid(score):
    return 1 / (1 + np.exp(-score))

# Fonction d'erreur
def compute_loss(vrai_label, prediction):
    epsilon = 1e-15
    # rester strictement entre 0 et 1 pour éviter log(0)
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    # Moyenne de toute les erreurs et (-) car le log d'un nombre entre 0 et 1 est toujours négatif
    return -np.mean(vrai_label * np.log(prediction) + (1 - vrai_label) * np.log(1 - prediction))

def train_logistic_regression(images, labels, learning_rate=0.01, n_iterations=1000):
    nb_images, nb_features = images.shape
    
    # Initialiser les poids à zéro
    # formule de la regression logistic : z=X⋅weights+bias
    weights = np.zeros(nb_features)
    bias = 0
    
    for i in range(n_iterations):
        # Prédiction
        score = np.dot(images, weights) + bias
        # Probabilité de la prédiction
        predictions = sigmoid(score)
        
        # Ont prend la transposé des matrice pour avoir 2 matrice de même dimenssion
        weights_gradient = (1 / nb_images) * np.dot(images.T, (predictions - labels)) # Addition des erreurs 
        bias_gradient = (1 / nb_images) * np.sum(predictions - labels) # correstion des poids
        
        # Mise à jour des poids
        weights -= learning_rate * weights_gradient
        bias -= learning_rate * bias_gradient
        
        # Apprentissage du modèle, plus le nb descend plus le modèle apprend de ses errers
        if i % 100 == 0:
            loss = compute_loss(labels, predictions) # Modèle = juste ? Eloigné ? 
            print(f"Iteration {i}: Loss = {loss:.4f}")
    
    return weights, bias

# Connaissance du modèle
weights, bias = train_logistic_regression(X_valid, y_valid_encoded)

# Prédiction du modèle
def predict(images, weights, bias):
    score = np.dot(images, weights) + bias # score
    predictions = sigmoid(score) # Proba
    return np.round(predictions)  # Arondi

predictions = predict(X_test, weights, bias)
