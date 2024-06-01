import pandas as pd

def get_ihdp(nums=[1]):
    # https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP
    # The IHDP dataset consists of 747 samples with 25 covariates
    # There are 10 IHDP datasets, each with a different treatment effect generated from the same data generating process
    # nums is the number of the 10 IHDP datasets used to build the dataset we use 
    url_prefix = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/IHDP/csv/ihdp_npci_"
    url_suffix = ".csv"
    cols = ['treatment', 'y_factual', 'y_cfactual', 'mu0', 'mu1'] + ['x' + str(i) for i in range(1,26)]

    datasets = []

    for num in nums:
        url = url_prefix + str(num) + url_suffix
        data = pd.read_csv(url, names=cols)
        datasets.append(data)

    data = pd.concat(datasets)
    instance = {}
    instance['data'] = data
    instance['y'] = data['y_factual'].values
    # Normalize the y values
    instance['y'] = (instance['y'] - instance['y'].min()) / (instance['y'].max() - instance['y'].min())
    instance['name'] = 'IHDP'
    instance['group'] = data['x7'].values # This is gender
    instance['X'] = data.drop(['y_factual', 'y_cfactual', 'mu0', 'mu1'], axis=1).values
    instance['n'] = len(instance['y'])

    return instance 