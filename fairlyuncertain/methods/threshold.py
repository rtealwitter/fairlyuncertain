from fairlearn.postprocessing import ThresholdOptimizer
import warnings
import xgboost as xgb

def threshold_algorithm(instance, constraints):
    assert constraints in ['demographic_parity', 'equalized_odds'], 'Invalid constraints'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = ThresholdOptimizer(
            estimator=xgb.XGBClassifier(), constraints=constraints, predict_method='predict', prefit=False
        ).fit(
            instance['X_train'], instance['y_train'], sensitive_features=instance['group_train']
        ).predict(
            instance['X_test'], sensitive_features=instance['group_test']
        )
    return {'pred' : pred}

def threshold_sp(instance):
    return threshold_algorithm(instance, 'demographic_parity')

def threshold_eo(instance):
    return threshold_algorithm(instance, 'equalized_odds')