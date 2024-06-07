import fairlyuncertain as fu

for is_binary in [True, False]:
    datasets = fu.binary_datasets if is_binary else fu.regression_datasets
    algorithm_names = fu.binary_uncertainty if is_binary else fu.regression_uncertainty

    algorithms = {algo_name: fu.algorithms[algo_name] for algo_name in algorithm_names}

    results = fu.get_calibration_table_data(is_binary=is_binary, algorithms=algorithms, datasets=datasets, num_runs=10, folder='cached')

    type = 'binary' if is_binary else 'regression'
    filename = f'tables/nll_{type}.tex'
    fu.print_table(results, filename=filename)