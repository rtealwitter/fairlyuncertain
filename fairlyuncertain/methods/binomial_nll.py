import numpy as np

def binomial_nll(instance, Model):
    model = Model('classifier', instance)
    model.fit(instance['X_train'], instance['y_train'])
    p = model.predict_proba(instance['X_test'])[:,1]
    pred = model.predict(instance['X_test'])
    std = np.sqrt(p * (1-p))
    return {'pred': pred, 'std': std, 'p': p}