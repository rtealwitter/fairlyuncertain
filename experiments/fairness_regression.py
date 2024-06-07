import fairlyuncertain as fu
import pickle
from tqdm import tqdm


# Algorithms
algorithms = {algo_name: fu.algorithms[algo_name] for algo_name in fu.regression_fairness}

# # # PLOT # # #

filename = 'cached/regression_fairness_plot.pkl'

if not os.path.exists(filename):
    results = {}
    for dataset in tqdm(fu.regression_datasets):
        instance = fu.load_instance(dataset)
        results[dataset] = {'instance' : instance}
        for algorithm in algorithms:
            results[dataset][algorithm] = algorithms[algorithm](instance)
    pickle.dump(results, open(filename, 'wb'))

results = pickle.load(open(filename, 'rb'))

plot_algorithms = ['True', 'Baseline', 'Normal NLL']

fu.plot_regression_fairness(results, fu.regression_datasets, plot_algorithms)

# # # TABLE # # #

filename = 'cached/regression_fairness.pkl'

if not os.path.exists(filename):
    num_runs = 10
    results = {}
    for dataset in tqdm(fu.regression_datasets):
        results[dataset] = {algo : [] for algo in algorithms}
        for _ in range(num_runs):
            instance = fu.load_instance(dataset) 
            for algo_name, algorithm in algorithms.items():
                output = algorithm(instance)
                sp = fu.get_regression_statistical_parity(instance['y_test'], instance['group_test'], output)
                results[dataset][algo_name] += [sp]

        pickle.dump(results, open(filename, 'wb'))
    
results = pickle.load(open(filename, 'rb'))


fu.print_table(results, 'tables/regression_fairness.tex')
