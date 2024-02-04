
def loss_are(y_test: np.array, y_pred: np.array, phi: float=5e-4, ordre: int=2) -> np.array:
    """
    Calcule la fonction de perte ARE, basée sur l'erreur relative absolue et un seuil.

    Paramètres:
      - y_test (np.array): Le tableau des valeurs réelles.
      - y_pred (np.array): Le tableau des valeurs prédites.
      - phi (float): Le seuil au-delà duquel une erreur est considérée comme significative.
      - ordre (int): L'ordre de la fonction de perte.

    Retourne:
      - np.array: Un tableau contenant les valeurs de la fonction de perte pour chaque prédiction.
    """
    
    return (np.abs(y_pred - y_test)**ordre/(y_test**ordre + 1e-6) > phi).astype(int)


def loss_ape(y_test: np.array, y_pred: np.array, ordre: int=2) -> np.array:
    """
    Calcule la fonction de perte APE, basée sur l'erreur absolue en pourcentage.
    
    Paramètres:
      - y_test (np.array): Le tableau des valeurs réelles.
      - y_pred (np.array): Le tableau des valeurs prédites.
      - ordre (int): L'ordre de la fonction de perte.

    Retourne:
      - np.array: Un tableau contenant les valeurs de la fonction de perte pour chaque prédiction.
    """
    
    res = (np.abs(y_pred - y_test))
    res = res**ordre/(np.max(res)**ordre)
    return res