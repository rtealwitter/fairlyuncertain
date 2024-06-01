import numpy as np
import scipy

rng = np.random.default_rng()
def compute_p_value(binary_preds):
    nA = np.sum(binary_preds==0, axis=0)
    nB = np.sum(binary_preds==1, axis=0)
    p_values = np.array([scipy.stats.binomtest(nA[i], nA[i]+nB[i], p=0.5).pvalue for i in range(binary_preds.shape[1])])
    return p_values

def selective_ensemble(instance, Model, num_ensemble=10):
    preds = []
    n = len(instance['X_train'])
    for i in range(num_ensemble):
        indices = rng.choice(n, n, replace=True)
        X = instance['X_train'][indices]
        y = instance['y_train'][indices]
        # HERE
        model = Model('classifier', instance)
        model.fit(X, y)
        preds.append(model.predict(instance['X_test']))
    preds = np.array(preds)
    p_values = compute_p_value(preds)
    mode = scipy.stats.mode(preds, axis=0)[0].squeeze()
    return {'pred': mode, 'std': p_values}