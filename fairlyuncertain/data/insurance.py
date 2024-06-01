import pandas as pd

def get_insurance():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/ff933238d9c30da179b7e3ad4b6ca938d67efc53/insurance.csv"
    data = pd.read_csv(url)
    instance = {}
    # Replace the categorical values with numerical values
    categorical_columns = ['sex', 'region', 'smoker']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    # Drop na values
    data = data.dropna()
    instance['data'] = data
    instance['name'] = 'insurance'
    y = data['charges']
    # Normalize the charges
    y = (y - y.min()) / (y.max() - y.min())
    instance['y'] = y.values
    instance['group'] = data['sex_male'].astype(int).values
    instance['X'] = (data.drop(columns=['charges'])).astype(float).values
    instance['n'] = data.shape[0]
    return instance
