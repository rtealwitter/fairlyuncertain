import os
# Add the path to the parent directory to augment search for module
os.sys.path.append(os.path.dirname(os.path.abspath('.')))
import fairlyuncertain as fu

for is_binary in [True, False]:
    datasets = fu.binary_datasets if is_binary else fu.regression_datasets
    algorithm_names = fu.binary_uncertainty if is_binary else fu.regression_uncertainty

    algorithms = {algo_name: fu.algorithms[algo_name] for algo_name in algorithm_names}

    results = fu.get_calibration_plot_data(algorithms=algorithms, datasets=datasets)
    fu.plot_calibration(results=results, is_binary=is_binary, algorithms = algorithms, datasets=datasets)