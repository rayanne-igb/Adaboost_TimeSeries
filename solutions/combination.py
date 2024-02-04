
def weighted_median(predictions: list, weights: list) -> float:
    """
    Calcule la médiane pondérée d'une liste à l'aide des poids qui y sont appliqués.

    Paramètres :
        - predictions (list) : La liste des prédictions pondérées.
        - weights (list) : La liste des poids associés à chaque prédiction.

    Retourne :
        - L'indice de la médiane.
    """
    median = np.sort(predictions)[len(predictions)//2]
    index_median = np.argsort(predictions)[len(predictions)//2]
    
    true_median = median/weights[index_median]   
    return true_median


def combination(learners: list, learners_weight: list, X: np.array, combination_method: str = 'weighted_mean') -> np.array:
    """
    Combine les prédictions de plusieurs apprenants selon une méthode spécifiée, avec ou sans pondération.

    Paramètres :
        - learners (list): Liste des modèles à combiner. 
        - learners_weight (list): Liste des poids associés à chaque modèle. Utilisé uniquement pour les méthodes pondérées.
        - X (np.array): Données d'entrée sur lesquelles les prédictions sont effectuées.
        - combination_method (str, optionnel): Méthode utilisée pour combiner les prédictions.
            Valeurs possibles : 'weighted_mean', 'mean', 'weighted_median', 'median'. Par défaut à 'weighted_mean'.

    Retourne :
        - np.array: La prédiction combinée pour les données d'entrée `X`, calculée selon la méthode spécifiée.
    """
        
    if combination_method=='weighted_mean':
        beta_sum = np.sum(learners_weight)
        pred = [learners[i].predict(X) * learners_weight[i]/beta_sum for i in range(len(learners))]
        pred = np.array(pred)
        
        y_pred = np.sum(pred, axis=0)
        
    elif combination_method=='mean':
        pred = [learners[i].predict(X) for i in range(len(learners))]
        pred = np.array(pred)
        
        y_pred = np.mean(pred, axis=0)
        
    elif combination_method=='weighted_median':
        pred = [learners[i].predict(X) * learners_weight[i] for i in range(len(learners))]
        pred = np.array(pred)
 
        y_pred = np.array([weighted_median(pred[:,i], learners_weight) for i in range(pred.shape[1])])
        
    elif combination_method=='median':
        pred = [learners[i].predict(X) for i in range(len(learners))]
        pred = np.array(pred)
        
        y_pred = np.median(pred, axis=0)
        
    else:
        raise ValueError("La méthode de combinaison spécifiée n'est pas valide.")
    
    return y_pred