def get_aif360(name):
    from aif360.datasets import AdultDataset, BankDataset, CompasDataset, GermanDataset, LawSchoolGPADataset, MEPSDataset19

    aif360_datasets = {
        'Adult' : {'loader' : AdultDataset, 'protected_attribute_names' : ['sex'], 'privileged_classes' : [['Male']], 'categorical_features' : ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'race']}, 
        'Bank' : {'loader' : BankDataset, 'protected_attribute_names' : ['age'], 'privileged_classes' : [lambda x : x >= 40], 'categorical_features' : ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']},
        'COMPAS' : {'loader' : CompasDataset, 'protected_attribute_names' : ['sex'], 'privileged_classes' : [['Male']], 'categorical_features' : ['age_cat', 'race', 'c_charge_degree', 'c_charge_desc']},
        'German' : {'loader' : GermanDataset, 'protected_attribute_names' : ['age'], 'privileged_classes' : [lambda x : x >= 25], 'categorical_features': ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property', 'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker'], 'features_to_drop':['personal_status', 'sex']},
        'MEPS' : {'loader' : MEPSDataset19, 'protected_attribute_names' : ['RACE'], 'privileged_classes' : [['White']],},
        'Law School' : {'loader' : LawSchoolGPADataset, 'protected_attribute_names' : ['race', 'gender'], 'privileged_classes' : [['white'], ['male']], 'categorical_features' : []},
    }
    assert name in aif360_datasets.keys(), f"Dataset {name} not one of {list(aif360_datasets.keys())}."

    if name in ['German', 'MEPS']:
        data, description = aif360_datasets[name]['loader'](
            protected_attribute_names=aif360_datasets[name]['protected_attribute_names'],
            privileged_classes=aif360_datasets[name]['privileged_classes'],
            features_to_drop=aif360_datasets[name].get('features_to_drop', [])
        ).convert_to_dataframe()
    else:
        data, description = aif360_datasets[name]['loader'](
            protected_attribute_names=aif360_datasets[name]['protected_attribute_names'],
            privileged_classes=aif360_datasets[name]['privileged_classes'],
            categorical_features=aif360_datasets[name]['categorical_features'],
        ).convert_to_dataframe()
    # Remove for pickling
    aif360_datasets[name].pop('loader')
    aif360_datasets[name].pop('privileged_classes')
    aif360_datasets[name]['description'] = description
    label_name = description['label_names'][0]
    protected_attribute_name = description['protected_attribute_names'][0]
    y = data[label_name].values
    X = data.drop(columns=label_name).values
    aif360_datasets[name]['X'] = X
    aif360_datasets[name]['y'] = y
    aif360_datasets[name]['n'] = X.shape[0]
    aif360_datasets[name]['group'] = data[protected_attribute_name].values
    if name in ['German']:
        aif360_datasets[name]['y'] -= 1
    
    return aif360_datasets[name]
