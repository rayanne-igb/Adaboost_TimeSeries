
def ts_to_windows(ts: np.array, size_window: int = 18):
    """
    Transforme une série temporelle en ensembles de données pour l'apprentissage supervisé 
    en utilisant la méthode des fenêtres glissantes.

    Paramètres:
        - ts (np.array): Le tableau contenant les valeurs de la série temporelle. 
        - size_window (int, optionnel): La taille de la fenêtre glissante, ie le nombre 
                                        de valeurs consécutives de la série temporelle à utiliser 
                                        comme une seule entrée.
                                        La valeur par défaut est 18 car 18 observations sont à prédire

    Retourne:
        - Tuple de deux éléments:
            - X (np.array) : Un tableau numpy de taille `(nb_windows, size_window)`, où chaque ligne 
                    représente une fenêtre de `size_window` valeurs consécutives de la série temporelle.
            - y (np.array) : Un tableau numpy de forme `(nb_windows,)`, contenant la valeur suivant 
                    chaque fenêtre dans `X`, servant de sortie correspondante.

    """
    X = []
    y = []
    
    for i in range(len(ts) - size_window):
        X.append(ts[i:i + size_window])
        y.append(ts[i + size_window])
        
    return np.array(X),np.array(y)