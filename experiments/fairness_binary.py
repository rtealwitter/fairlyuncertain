# Add the path to the parent directory to augment search for module
import os
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
import fairlyuncertain as fu
from tqdm import tqdm
import pickle
# reload
from importlib import reload
reload(fu)

algorithms = {k: fu.algorithms[k] for k in fu.binary_fairness}

num_runs = 10
    
for dataset in tqdm(fu.binary_datasets):
    filename = f'cached/binary_fairness_{dataset}.pkl'
    if os.path.exists(filename): continue
    results = {}
    for num_run in range(num_runs):
        instance = fu.load_instance(dataset)
        results[num_run] = {'instance' : instance}
        for algo_name, algorithm in algorithms.items():
            results[num_run][algo_name] = algorithm(instance)
    pickle.dump(results, open(filename, 'wb'))

for dataset in fu.binary_datasets:
    filename = f'cached/binary_fairness_{dataset}.pkl'
    results = pickle.load(open(filename, 'rb'))

    metric_values = fu.compute_binary_fairness(results, algorithms, fu.binary_metrics)

    fu.print_table(metric_values, filename=f'tables/binary_fairness_{dataset}.tex')