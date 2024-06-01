import fairlearn.reductions as red
import xgboost as xgb
import numpy as np
import warnings

moments = {
    'SP' : lambda : red.DemographicParity(),
    'EO' : lambda : red.EqualizedOdds(),
    #'ERP' : lambda : red.ErrorRateParity(),
    'Square' : lambda : red.BoundedGroupLoss(red.SquareLoss(min_val=-np.inf, max_val=np.inf), upper_bound=0.1),
    'Absolute' : lambda : red.BoundedGroupLoss(red.AbsoluteLoss(min_val=-np.inf, max_val=np.inf), upper_bound=0.1),
}

def exponentiated_gradient_algorithm(instance, constraints):
    is_binary = len(np.unique(instance['y_train'])) == 2
    model = xgb.XGBClassifier() if is_binary else xgb.XGBRegressor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = red.ExponentiatedGradient(
            model, constraints=constraints,
        ).fit(
            instance['X_train'], instance['y_train'], sensitive_features=instance['group_train']
        ).predict(instance['X_test'])
    return {'pred' : pred}

def gridsearch_algorithm(instance, constraints):
    model = red.GridSearch(
        xgb.XGBRegressor(), constraints=constraints,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            instance['X_train'], instance['y_train'], sensitive_features=instance['group_train']
        )
        pred = model.predict(instance['X_test'])

    return {'pred' : pred}

def exponentiated_gradient(instance, moment_method):
    if moment_method not in moments:
        raise ValueError(f'Invalid moment method. Must be in {moments.keys()}')
    return exponentiated_gradient_algorithm(instance, moments[moment_method]())

def gridsearch(instance, moment_method):
    if moment_method not in moments:
        raise ValueError(f'Invalid moment method. Must be in {moments.keys()}')
    if moment_method in ['SP', 'EO', 'ERP']:
        output = gridsearch_algorithm(instance, moments[moment_method]())
        return {'pred': np.round(output['pred'], 0)}
    return gridsearch_algorithm(instance, moments[moment_method]())
