import fairlyuncertain as fu
from tqdm import tqdm

max_depths = list(range(5, 16))

num_runs = 0

for is_binary in [True, False]:
    type = 'binary' if is_binary else 'regression'
    datasets = fu.binary_datasets if is_binary else fu.regression_datasets
    algorithm_names = fu.binary_uncertainty if is_binary else fu.regression_uncertainty
    algorithms = {algo_name: fu.algorithms[algo_name] for algo_name in algorithm_names}

    for i in range(num_runs):    
        results = {}
        for dataset in tqdm(datasets):
            results[dataset] = fu.get_consistency_data(dataset, algorithms, max_depths)        

    results = {}
    for dataset in tqdm(datasets):
        results[dataset] = fu.get_consistency_data(dataset, algorithms, max_depths)
    
    fu.plot_consistency(results, is_binary, algorithms, datasets, folder='figures')

    table = fu.get_consistency_table(results, datasets, algorithms)
    filename = 'tables/consistency_' + type + '.tex'

    fu.print_table(table, filename)