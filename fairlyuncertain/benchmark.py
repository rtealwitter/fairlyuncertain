from .data import load_instance
from .utils import get_nll
import numpy as np
import os

# # # CALIBRATION # # #

def get_calibration_plot_data(algorithms, datasets):
    results = {}
    for dataset in datasets:
        instance = load_instance(dataset)
        results[dataset] = {'instance' : instance}
        for algorithm_name, algorithm in algorithms.items():
            results[dataset][algorithm_name] = algorithm(instance)
    return results

def load_calibration_results(filename, algorithms, datasets):
    results = {dataset : {algo_name : [] for algo_name in algorithms} for dataset in datasets}
    with open(filename, 'r') as f:
        for line in f:
            saved = eval(line)
            for algo_name in algorithms:
                if algo_name == 'dataset': continue
                results[saved['dataset']][algo_name].append(saved[algo_name])
    return results

def get_calibration_table_data(is_binary, algorithms, datasets, num_runs=10, folder=None):
    if folder is not None:
        # Make folder if it does not exist
        if not os.path.exists(folder):
            os.makedirs(folder)
        type = 'binary' if is_binary else 'regression'
        filename = folder + f'/calibration_{type}.csv'
        if os.path.exists(filename):
            return load_calibration_results(filename, algorithms, datasets) 
    results = {}

    for dataset in datasets:
        results[dataset] = {algo_name : [] for algo_name in algorithms}
        for _ in range(num_runs):
            instance = load_instance(dataset)
            for algo_name, algorithm in algorithms.items():
                output = algorithm(instance)
                nll = get_nll(output['pred'], output['std'], instance['y_test'], is_binary).mean()
                results[dataset][algo_name].append(nll)
            if folder is not None:
                saved = {algo_name : results[dataset][algo_name][-1] for algo_name in algorithms}
                saved['dataset'] = dataset
                with open(filename, 'a') as f:
                    if 'nan' not in str(saved):
                        f.write(str(saved) + '\n')

    if folder is not None:
        results = load_calibration_results(filename, algorithms, datasets)

    return results

# # # CONSISTENCY # # #

def load_consistency_results(filename, algorithms):
    results = {name: {} for name in algorithms}
    if not os.path.exists(filename):
        return results    
    with open(filename, 'r') as f:
        for line in f:
            saved = eval(line)
            max_depth = saved['max_depth']
            for name in saved:
                if name == 'max_depth': continue
                if max_depth not in results[name]:
                    results[name][max_depth] = []
                results[name][max_depth] += [saved[name]]
    return results

def get_consistency_data(dataset, algorithms, max_depths):
    filename = f'cached/consistency_{dataset}.csv'
    instance = load_instance(dataset)
    for max_depth in max_depths:
        
        instance['max_depth'] = max_depth
        saved = {'max_depth': max_depth}
        for algo_name, algorithm in algorithms.items():
            output = algorithm(instance)
            saved[algo_name] = list(output['std'])
        if len(saved) > 1:
            with open(filename, 'a') as f:
                f.write(str(saved) + '\n')
    
    return load_consistency_results(filename, algorithms)

# # # BINARY FAIRNESS # # #

def compute_binary_fairness(results, algorithms, metrics): 
    metric_values = {metric : {algo_name : [] for algo_name in algorithms} for metric in metrics}
    metric_values[r'Included \%'] = {algo_name : [] for algo_name in algorithms}

    for num_run in results:
        instance = results[num_run]['instance']
        y = instance['y_test']
        group = instance['group_test'] == 1
        for algo_name in algorithms:
            pred = results[num_run][algo_name]['pred']
            # Gridsearch returns floats
            pred = pred.clip(0, 1)
            pred = np.round(pred)
            assert np.all(np.unique(pred)==np.array([0,1])), 'Predictions are not binary'
            # No abstaining
            if 'std' not in results[num_run][algo_name]:
                for metric in metrics:
                    val = metrics[metric](pred, y, group, run_checks=False)
                    metric_values[metric][algo_name].append(val)
                metric_values[r'Included \%'][algo_name].append(100)
                continue
            # Abstaining at best percentile for SP
            percentiles = np.arange(75, 101, 1)
            std = results[num_run][algo_name]['std'] 
            best_val = np.inf
            for percentile in percentiles:
                include = std <= np.percentile(std, percentile)
                val = 0
                for metric in metrics:
                    normalization = metrics[metric](results[num_run]['Baseline']['pred'], y, group, run_checks=False)
                    metric_val = metrics[metric](pred[include], y[include], group[include], run_checks=False)
                    val += metric_val / normalization
                if val < best_val:
                    best_val = val
                    best_percentile = percentile
            include = std <= np.percentile(std, best_percentile) 
            for metric in metrics:
                metric_val = metrics[metric](pred[include], y[include], group[include], run_checks=False)
                metric_values[metric][algo_name].append(metric_val)
            metric_values[r'Included \%'][algo_name].append(int(100 * include.mean()))

    return metric_values