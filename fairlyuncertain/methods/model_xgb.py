import scipy.stats
import torch
import numpy as np
import scipy
import xgboost as xgb
from torch.utils.data import DataLoader, Dataset

seed = 0
rng = np.random.default_rng(seed=seed)

def normal_nll(pred, dtrain):
    y = dtrain.get_label().reshape(-1, 2)[:,0]
    mu, sigma= pred[:,0], pred[:,1]
    # L = 2*np.log(sigma) + 0.5 * ((mu - y) / sigma) ** 2

    dLdmu = (mu- y) / sigma**2
    dLdsigma = -(mu- y)**2 / sigma**3 + 2 * sigma
    dLdmu2 = 1 / sigma**2
    dLdsigma2 = 3 * (mu - y)**2 / sigma**4 + 2

    grad = np.column_stack([dLdmu, dLdsigma]).reshape(-1, 1)

    hess = np.column_stack([dLdmu2, dLdsigma2]).reshape(-1, 1)

    return grad, hess

def beta_nll(pred, dtrain):
    y = dtrain.get_label().reshape(-1, 2)[:,0]
    mu, sigma= pred[:,0], pred[:,1]
    beta = .5
    weighting = sigma**(2*beta)
    # L = (2*np.log(sigma) + 0.5 * ((mu - y) / sigma) ** 2) * weighting

    dLdmu = (mu- y) / sigma**2 * weighting
    dLdsigma = (-(mu- y)**2 / sigma**3 + 2 * sigma) * weighting
    dLdmu2 = 1 / sigma**2 * weighting
    dLdsigma2 = (3 * (mu - y)**2 / sigma**4 + 2) * weighting

    grad = np.column_stack([dLdmu, dLdsigma]).reshape(-1, 1)

    hess = np.column_stack([dLdmu2, dLdsigma2]).reshape(-1, 1)

    return grad, hess
    
def faithful_nll(pred, dtrain):
    y = dtrain.get_label().reshape(-1, 2)[:,0]
    mu, sigma= pred[:,0], pred[:,1]
    # L = 2*np.log(sigma) + 0.5 * ((mu - y) / sigma) ** 2

    dLdmu = (mu- y)
    dLdsigma = -(mu- y)**2 / sigma**3 + 2 * sigma
    dLdmu2 = np.ones_like(dLdmu)
    dLdsigma2 = 3 * (mu - y)**2 / sigma**4 + 2

    grad = np.column_stack([dLdmu, dLdsigma]).reshape(-1, 1)

    hess = np.column_stack([dLdmu2, dLdsigma2]).reshape(-1, 1)

    return grad, hess


xgb_multi_losses = {
    'Normal NLL' : normal_nll,
    r'$\beta$-NLL' : beta_nll,
    'Faithful NLL' : faithful_nll,
}

def xgb_train(loss_name, xgb_instance):
    y = np.column_stack([xgb_instance['y_train'], np.zeros_like(xgb_instance['y_train'])])
    dtrain = xgb.DMatrix(xgb_instance['X_train'], label=y)
    bst = xgb.train(
        {
            'tree_method': 'hist',
            'max_depth' : xgb_instance['max_depth'],
            'disable_default_eval_metric' : True,
            'multi_strategy' : 'multi_output_tree',
        },
        dtrain = dtrain,
        num_boost_round = xgb_instance['num_boost_round'],
        obj = xgb_multi_losses[loss_name],
    )
    return lambda x: bst.predict(xgb.DMatrix(x)).squeeze()

class Model:
    def __init__(self, loss_name=None, instance=None):
        if instance is None:
            instance = {'max_depth' : 6, 'num_boost_round' : 10}
        self.max_depth = instance['max_depth']
        self.num_boost_round = instance['num_boost_round']
        self.loss_name = loss_name
        self.model = None
        if loss_name == 'classifier':
            self.model = xgb.XGBClassifier(max_depth=self.max_depth, n_estimators=self.num_boost_round)
        elif loss_name == 'regression':
            self.model = xgb.XGBRegressor(max_depth=self.max_depth, n_estimators=self.num_boost_round)
    
    def fit(self, X, y, sample_weight=None):
        if self.loss_name in ['classifier', 'regression']:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model = xgb_train(self.loss_name, {'X_train': X, 'y_train': y, 'max_depth': self.max_depth, 'num_boost_round': self.num_boost_round})
    
    def predict(self, X):
        if self.loss_name in ['classifier', 'regression']:
            return self.model.predict(X)
        return self.model(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
