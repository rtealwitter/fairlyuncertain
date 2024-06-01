from folktables import ACSDataSource, ACSEmployment, ACSIncome, ACSPublicCoverage

import pandas as pd
import numpy as np

data_types = {
    "AGEP": {
        "type":"ordinal"
    },
    "ANC": {
        "type":"categorical"
    },
    "CIT": {
        "type":"categorical"
    },
    "COW": {
        "type":"categorical"
    },
    "DEAR": {
        "type":"categorical"
    },
    "DEYE": {
        "type":"categorical"
    },
    "DIS": {
        "type":"categorical"
    },
    "DREM": {
        "type":"categorical"
    },
    "ESP": {
        "type":"categorical"
    },
    "ESR": {
        "type":"categorical"
    },
    "FER": {
        "type":"categorical"
    },
    "JWTR": {
        "type":"categorical"
    },
    "MAR": {
        "type":"categorical"
    },
    "MIG": {
        "type":"categorical"
    },
    "MIL": {
        "type":"categorical"
    },
    "NATIVITY": {
        "type":"categorical"
    },
    "OCCP": {
        "type":"categorical"
    },
    "PINCP": {
        "type":"categorical"
    },
    "POBP": {
        "type":"categorical"
    },
    "POVPIP": {
        "type":"numerical"
    },
    "POWPUMA": {
        "type":"categorical"
    },
    "PUMA": {
        "type":"categorical"
    },
    "RAC1P": {
        "type":"categorical"
    },
    "GCL": {
        "type":"categorical"
    },
    "RELP": {
        "type":"categorical"
    },
    "SCHL": {
        "type":"categorical"
    },
    "SEX": {
        "type":"categorical"
    },
    "ST": {
        "type":"categorical"
    },
    "WKHP": {
        "type":"ordinal"
    },
    "PUBCOV": {
        "type":"categorical"
    },
    "JWMNP": {
        "type":"categorical"
    },
}

def get_categorical_features(scenario, data_types):
    categorical_features = []
    for feature in scenario._features:
        if data_types[feature]['type'] == 'categorical':
            categorical_features.append(feature)
    return categorical_features

ACS_Scenarios = {
    "ACSEmployment": {'loader': ACSEmployment, 
                      'categorical_features' : get_categorical_features(ACSEmployment, data_types),
                      'sensitive_features': [1, 2]
                      },
    "ACSIncome": {'loader': ACSIncome, 
                      'categorical_features' : get_categorical_features(ACSIncome, data_types),
                      'sensitive_features': [1, 2]
                 },
    "ACSPublicCoverage": {'loader': ACSPublicCoverage, 
                          'categorical_features' : get_categorical_features(ACSPublicCoverage, data_types),
                          'sensitive_features': [1, 2]
                         }
}

def return_acs_data_scenario(acs_data, scenario="ACSEmployment", verbose=False):
    scenario = ACS_Scenarios[scenario]['loader']
    features, label, group = scenario.df_to_numpy(acs_data)
    if verbose:
        print(features, label, group)

    np_all_data = np.c_[features,label]

    df = pd.DataFrame(np_all_data, columns = scenario._features + [scenario._target])
    features = pd.DataFrame(features, columns = scenario._features)
    target = pd.DataFrame(label, columns = [scenario._target])
    group = pd.DataFrame(group, columns = [scenario._group])

    return df, features, target, group

def get_acs(name, states=['NY']):
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=states, download=True)
    _, features, target, group = return_acs_data_scenario(acs_data, scenario=name)

    # label_name = target.columns
    protected_attribute_name = group.columns
    y = target
    y = y.astype(int)
    X = features

    criteria = group[protected_attribute_name[0]].isin(ACS_Scenarios[name]['sensitive_features'])
    group_filtered = group[criteria]
    group_filtered = group_filtered - 1
    X = X[criteria]
    y = y[criteria]
    
    indices_subsample = np.random.choice(X.shape[0], int(X.shape[0]*0.1), replace=False)
    X = X.iloc[indices_subsample]
    y = y.iloc[indices_subsample]
    group_filtered = group_filtered.iloc[indices_subsample]

    ACS_Scenarios[name].pop('loader')
    ACS_Scenarios[name]['X'] = X.values
    ACS_Scenarios[name]['y'] = y.values.ravel()
    ACS_Scenarios[name]['n'] = X.shape[0]
    ACS_Scenarios[name]['group'] = group_filtered.values.ravel()
    ACS_Scenarios[name]['protected_attribute_names'] = protected_attribute_name
    
    return ACS_Scenarios[name]

    


  