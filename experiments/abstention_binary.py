from tqdm import tqdm
import os
import numpy as np
# Add the path to the parent directory to augment search for module
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
import fairlyuncertain as fu


algorithm_names = ['Random', 'Selective Ensemble', 'Self-(in)consistency', 'Binomial NLL']

algorithms = {algo_name: fu.algorithms[algo_name] for algo_name in algorithm_names}

percentiles = np.arange(50, 101, 1)

results= {}
for dataset in tqdm(fu.binary_datasets):
    instance = fu.load_instance(dataset)
    results[dataset] = {'instance' : instance}
    for algo_name in algorithms:
        results[dataset][algo_name] = algorithms[algo_name](instance)

for metric_name in fu.binary_metrics:
    fu.plot_abstention(results, algorithms, fu.binary_datasets, percentiles, fu.binary_metrics, metric_name=metric_name, folder='figures')