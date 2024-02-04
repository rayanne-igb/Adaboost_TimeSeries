
def beta_bounded(loss_hat: float) -> float:
    """
    Calcule le coefficient beta pour un critère d'arrêt borné.

    Paramètres :
        - loss_hat (float) : La perte estimée.

    Retourne :
        - Le coefficient beta calculé ou -1 si `loss_hat` > 0.5.
    """
        
    if loss_hat <= 0.5:
        return np.log((1 - loss_hat)/loss_hat)
    else:
        return -1
    
def beta_unbounded(loss_hat: float, ordre: int = 2) -> float:
    """
    Calcule le coefficient beta pour un critère d'arrêt non borné avec un ordre ajustable.

    Paramètres :
        - loss_hat (float) : La perte estimée.
        - puissance (int, optionnel): Le coefficient de puissance, par défaut à 2.

    Retourne :
        - Le coefficient beta calculé.
    """
    
    return np.log(1/(loss_hat**ordre))