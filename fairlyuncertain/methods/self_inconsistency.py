import numpy as np
import scipy

rng = np.random.default_rng()

def self_inconsistency_ensemble(instance, Model, B=10):
    # https://arxiv.org/pdf/2301.11562
    preds = []
    n = len(instance['X_train'])
    for i in range(B):
        indices = rng.choice(n, n, replace=True)
        X = instance['X_train'][indices]
        y = instance['y_train'][indices]
        # HERE
        model = Model('classifier', instance)
        model.fit(X, y)
        preds.append(model.predict(instance['X_test']))
    preds = np.array(preds)
    B1 = np.sum(preds, axis=0)
    B0 = B - B1
    sc = 2 * B0 * B1 / (B * (B - 1))
    mid1, mid2 = (B+1) // 2, B // 2
    max_val = 2 * mid1 * mid2 / (B * (B - 1))
    assert np.all(sc <= max_val), 'Self inconsistency score is greater than theoretical maximum.'
    mode = scipy.stats.mode(preds, axis=0)[0].squeeze()
    return {'pred': mode, 'std': sc}