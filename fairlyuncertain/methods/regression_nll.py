def normal_nll_wrap(instance, loss_name, Model):
    model = Model(loss_name, instance)
    model.fit(instance['X_train'], instance['y_train'])
    preds = model.predict(instance['X_test'])
    mu, sigma = preds[:,0], preds[:,1]
    return {'pred' : mu, 'std' : sigma}

def normal_nll(instance, Model):
    return normal_nll_wrap(instance, 'Normal NLL', Model)

def beta_nll(instance, Model):
    return normal_nll_wrap(instance, r'$\beta$-NLL', Model)

def faithful_nll(instance, Model):
    return normal_nll_wrap(instance, 'Faithful NLL', Model)