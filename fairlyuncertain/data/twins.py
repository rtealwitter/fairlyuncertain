import pandas as pd

def get_twins():
    # https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS
    # Twins dataset
    url_weight = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/TWINS/twin_pairs_T_3years_samesex.csv"

    url_X = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/TWINS/twin_pairs_X_3years_samesex.csv"

    url_y = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/TWINS/twin_pairs_Y_3years_samesex.csv"

    weight = pd.read_csv(url_weight) # birth weight
    X = pd.read_csv(url_X) # covariates
    y = pd.read_csv(url_y) # mortality (binary)

    data = pd.concat([weight, X, y], axis=1).drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
    # Drop data where nprevisitq is nan
    data = data.dropna(subset=['nprevistq'])
    # Drop columns with missing values
    data = data.dropna(axis=1)
    data['sensitive'] = (data['crace'] == 1).astype(int)

    instance = {}

    instance['data'] = data
    instance['group'] = data['sensitive'].values
    instance['y'] = data['nprevistq'].values
    # Normalize the y values
    instance['y'] = (instance['y'] - instance['y'].min()) / (instance['y'].max() - instance['y'].min())
    instance['X'] = data.drop(columns=['nprevistq']).values.astype(float)
    instance['n'] = instance['X'].shape[0]

    return instance
