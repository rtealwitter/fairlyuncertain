import numpy as np
from ucimlrepo import fetch_ucirepo 

def get_communities_and_crimes():
    instance = {}
    # fetch dataset 
    dataset = fetch_ucirepo(id=183) 

    dataset.data.features = dataset.data.features.replace('?', np.nan)
    # Drop columns with more than 10% missing values
    dataset.data.features.dropna(thresh=int(dataset.data.features.shape[0]), axis=1, inplace=True)
    dataset.data.features['sensitive'] = np.array((dataset.data.features['racepctblack'] > .2).astype(int))
    

    instance['data'] = dataset.data
    # Drop rows with missing 

    instance['y'] = np.array(dataset.data.targets)
    instance['group'] = dataset.data.features['sensitive'].values
    instance['n'] = len(dataset.data.features)

    cols_to_drop = ['state', 'communityname', 'fold']
    X = np.array(dataset.data.features.drop(cols_to_drop, axis=1).values)
    # Replace '?' with NaN
    X = np.where(X == '?', np.nan, X)
    # Convert to float
    X = X.astype(float)
    instance['X'] = X
    return instance


