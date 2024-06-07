import numpy as np
from .model_xgb import Model

from .baseline import baseline, true, random
from .ensemble import ensemble
from .selective_ensemble import selective_ensemble
from .self_inconsistency import self_inconsistency_ensemble
from .binomial_nll import binomial_nll
from .regression_nll import normal_nll, beta_nll, faithful_nll
from .threshold import threshold_sp, threshold_eo
from .reductions import gridsearch, exponentiated_gradient, moments
from .ft_transformer import FTTransformerModel

# # # # BINARY AND REGRESSION # # # #

algorithms = {
    'True' : true,
    'Baseline' : lambda instance : baseline(instance, Model),
    'Random' : lambda instance : random(instance, Model),
    'Ensemble' : lambda instance : ensemble(instance, Model),
    'Selective Ensemble' : lambda instance : selective_ensemble(instance, Model),
    'Self-(in)consistency' : lambda instance : self_inconsistency_ensemble(instance, Model),
    'Binomial NLL' : lambda instance : binomial_nll(instance, Model),
    'Normal NLL' : lambda instance : normal_nll(instance, Model),
    r'$\beta$-NLL' : lambda instance : beta_nll(instance, Model),
    'Faithful NLL' : lambda instance : faithful_nll(instance, Model),
    'Threshold Optimizer SP' : threshold_sp,
    'Threshold Optimizer EO' : threshold_eo,
}

for moment in moments:
    algorithms[f'Exponentiated Gradient {moment}'] = lambda instance, moment=moment : exponentiated_gradient(instance, moment)

for moment in moments:
    algorithms[f'Grid Search {moment}'] = lambda instance, moment=moment : gridsearch(instance, moment)

binary_uncertainty = ['Ensemble', 'Selective Ensemble', 'Self-(in)consistency', 'Binomial NLL']

binary_fairness = ['Baseline']
for algorithm in algorithms:
    if 'SP' in algorithm or 'EO' in algorithm or 'ERP' in algorithm:
        binary_fairness.append(algorithm)

binary_fairness += ['Random', 'Ensemble', 'Selective Ensemble', 'Self-(in)consistency', 'Binomial NLL']

regression_uncertainty = ['Ensemble', 'Normal NLL', r'$\beta$-NLL', 'Faithful NLL']

regression_fairness = ['True', 'Baseline']

for algorithm in algorithms:
    if 'Square' in algorithm or 'Absolute' in algorithm:
        regression_fairness.append(algorithm)

regression_fairness += ['Ensemble', 'Normal NLL', r'$\beta$-NLL', 'Faithful NLL']
