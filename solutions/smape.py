
def smape(y_pred: np.array, y_true: np.array) -> float:
    """
    Calcule l'erreur SMAPE entre une série de valeur réelle et une série de valeur prédite.

    Paramètres :
        - y_pred (np.array) : Valeurs prédites.
        - y_true (np.array) : Valeurs réelles.

    Retourne :
        - float : La valeur SMAPE entre les valeurs prédites et les valeurs réelles.
        
    """
    if len(y_pred) != len(y_true):
        raise ValueError("Les deux séries de valeurs doivent avoir la même longueur.")

    error = 0
    
    for i in range(len(y_true)):
        if (np.abs(y_pred[i]) + np.abs(y_true[i])) ==0:
            error += 0
        else:
            error += np.abs(y_pred[i] - y_true[i]) / ((np.abs(y_pred[i]) + np.abs(y_true[i]))/2)
    
    return 100 * error / len(y_true)