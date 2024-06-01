# Add the path to the parent directory to augment search for module
import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
import fairlyuncertain as fu
from tqdm import tqdm

max_depths = list(range(5, 16))

for is_binary in [True, False]:
    datasets = fu.binary_datasets if is_binary else fu.regression_datasets
    algorithm_names = fu.binary_uncertainty if is_binary else fu.regression_uncertainty
    algorithms = {algo_name: fu.algorithms[algo_name] for algo_name in algorithm_names}

    type = 'binary' if is_binary else 'regression'

    results = {}
    for dataset in tqdm(datasets):
        results[dataset] = fu.get_consistency_data(dataset, algorithms, max_depths)
    
    fu.plot_consistency(results, is_binary, algorithms, datasets)

    table = fu.get_consistency_table(results, datasets, algorithms)
    filename = 'tables/consistency_' + type + '.tex'

    fu.print_table(table, filename, is_constrained=True, include_var=False)