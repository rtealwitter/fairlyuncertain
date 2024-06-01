import numpy as np

rng = np.random.default_rng()

def ensemble(instance, Model, num_ensemble=10):
    models = []
    # Check if instance['y_train'] is binary
    is_binary = len(np.unique(instance['y_train'])) == 2
    n = len(instance['X_train'])
    for i in range(num_ensemble):
        indices = rng.choice(n, n, replace=True)
        X = instance['X_train'][indices]
        y = instance['y_train'][indices]
        model = Model('classifier' if is_binary else 'regression', instance)
        model.fit(X, y)
        models.append(model)
    
    model_predict = lambda model, X : model.predict(X) if not is_binary else model.predict_proba(X)[:,1]
    
    def predict(X):
        preds = np.array([model_predict(model, X) for model in models]).T
        return np.mean(preds, axis=1), np.std(preds, axis=1)
    
    mu, sigma = predict(instance['X_test'])
    return {'pred': mu, 'std': sigma}