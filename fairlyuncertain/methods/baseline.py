import numpy as np

def baseline(instance, Model):
    is_binary = len(np.unique(instance['y_train'])) == 2
    # HERE
    model = Model('classifier' if is_binary else 'regression', instance)
    model.fit(instance['X_train'], instance['y_train'])
    pred = model.predict(instance['X_test'])
    return {'pred': pred}

def true(instance):
    return {'pred' : instance['y_test']}

def random(instance, Model):
    output = baseline(instance, Model)
    output['std'] = np.random.rand(len(instance['y_test']))/2 # Random values between 0 and 0.5
    return output