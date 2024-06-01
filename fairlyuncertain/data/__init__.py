import pickle
import os
import numpy as np
import requests

from .aif360 import get_aif360
from .communities import get_communities_and_crimes
from .insurance import get_insurance
from .ihdp import get_ihdp
from .twins import get_twins
from .acs_data import get_acs

dataloaders = {
    'Adult' : lambda : get_aif360('Adult'),
    'Bank' : lambda : get_aif360('Bank'),
    'COMPAS' : lambda : get_aif360('COMPAS'),
    'German' : lambda : get_aif360('German'),
    'MEPS' : lambda : get_aif360('MEPS'),
    'Law School' : lambda : get_aif360('Law School'),
    'Communities' : get_communities_and_crimes,
    'Insurance' : get_insurance,
    'IHDP' : get_ihdp,
    'Twins' : get_twins,
    'ACS': lambda : get_acs('ACSEmployment'),
}

def cache_dataset(name):
    print(f'Caching {name} dataset...')
    url = f'https://raw.githubusercontent.com/rtealwitter/naturalexperiments/main/fairlyuncertain/data/cached/{name}.pkl'
    filename = __file__.replace('__init__.py', f'cached/{name}.pkl')
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        open(filename, 'wb').write(r.content)
    else:
        output = dataloaders[name]()
        pickle.dump(output, open(filename, 'wb')) 

def read_dataset(name):
    filename = __file__.replace('__init__.py', f'cached/{name}.pkl')
    return pickle.load(open(filename, 'rb'))

def load_instance(name, train_split=.8, max_depth=6, num_boost_round=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Check cached folder exists
    if not os.path.exists(__file__.replace('__init__.py', 'cached')):
        os.makedirs(__file__.replace('__init__.py', 'cached'))
    # Cache dataset if not already cached
    if not os.path.exists(__file__.replace('__init__.py', f'cached/{name}.pkl')):
        cache_dataset(name)    

    real_dataset = read_dataset(name)
    n = real_dataset['n']
    train_indices = np.random.choice(n, int(n*train_split), replace=False)
    test_indices = np.array(list(set(range(n)) - set(train_indices)))
    instance = {
        'name' : name,
        'n': real_dataset['n'],
        'synthetic': False,
        'X': real_dataset['X'],
        'y': real_dataset['y'],
        'group' : real_dataset['group'],        
        'train_indices' : train_indices,
        'group_train' : real_dataset['group'][train_indices],
        'group_test' : real_dataset['group'][test_indices],
        'X_train' : real_dataset['X'][train_indices],
        'X_test' : real_dataset['X'][test_indices],
        'y_train' : real_dataset['y'][train_indices],
        'y_test' : real_dataset['y'][test_indices],
        'max_depth' : max_depth,
        'num_boost_round' : num_boost_round,
    }
    return instance

binary_datasets = ['ACS', 'Adult', 'Bank', 'COMPAS', 'German'] # MEPS requires data agreement
regression_datasets = ['Communities', 'IHDP', 'Insurance', 'Law School', 'Twins']

datasets = binary_datasets + regression_datasets