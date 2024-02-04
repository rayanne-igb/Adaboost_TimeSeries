
def evaluate_id(id:str,verbose: bool=True, combination_method: str ='weighted_mean', base_model: str='CART', loss_method: str='ape', beta_method: str='unbounded'):
    """
    Applique et évalue le modèle de prévision pour une série temporelle et une combinaison de méthodes donnés.
    
    Paramètres :
        - id (str) : L'identifiant des données de la série temporelle à prévoir.
        - verbose (bool, optionnel) : Si True, le modèle affichera des messages de progression pendant l'entraînement.
        - combination_method (str, optionnel) : La méthode utilisée pour combiner les prédictions des apprenants de base dans le modèle d'ensemble.
        - base_model (str, optionnel) : Le type de modèle de base à utiliser dans l'ensemble.
        - loss_method (str, optionnel) : La fonction de perte utilisée par le modèle.
        - beta_method (str, optionnel) : La méthode utilisée pour calculer les poids (bêtas) dans l'algorithme AdaBoost.
        
    Retourne :
        - Tuple de deux éléments :
            - pred (np.array) : Les prédictions pour l'ensemble de test.
            - smape (float) : L'erreur SMAPE entre les prédictions et les valeurs réelles de l'ensemble de test.
    """
    size_window = 18

    ts = get_ts(id)    
    ts_train = ts[:-18]
    ts_test = ts[-(18 + size_window):]
    X_train, y_train = ts_to_windows(ts_train, size_window=size_window)
    X_test, y_test = ts_to_windows(ts_test, size_window=size_window)
    
    model = MyAdaBoostForest(verbose=verbose, n_steps=50, combination=combination_method, base_model=base_model, loss=loss_method, beta=beta_method)
    model.fit(X_train, y_train)
    pred = model.predict_model(X_test)

    return pred, smape(pred, y_test)