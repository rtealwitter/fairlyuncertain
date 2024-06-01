import matplotlib.pyplot as plt
import numpy as np
from .utils import compute_cdf

plt.rcParams.update({'font.size': 14})

linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (1, (3, 5, 1, 5)), (2, (3, 5, 1, 5))]


# # # CALIBRATION # # #

def plot_calibration(results, is_binary, algorithms, datasets, folder='figures'):
    fig, axs = plt.subplots(1, len(datasets), figsize=(20, 4))

    markers = ['o', 's', 'v', 'D', 'P', 'X', 'H']

    for i, dataset in enumerate(datasets):
        instance = results[dataset]['instance']
        y = instance['y_test']
        for j, algo_name in enumerate(algorithms):
            std = results[dataset][algo_name]['std']
            if not is_binary:
                mean = results[dataset][algo_name]['pred']
            sigma_bins = np.linspace(np.percentile(std, 0), np.percentile(std, 100), 100)
            pred_std, empirical_std = [], []
            for k in range(len(sigma_bins)-1):
                criteria = (std >= sigma_bins[i]) & (std < sigma_bins[k+1])
                if np.sum(criteria) < 20: continue
                pred_std.append(np.mean(std[criteria]))
                if not is_binary:
                    y_centered = y[criteria] - mean[criteria]
                    empirical_std.append(np.std(y_centered))
                else:
                    empirical_std.append(np.std(y[criteria]))
            axs[i].plot(pred_std, empirical_std, marker=markers[j], label=algo_name)
            
        # get xlim and ylim
        min_val = min(axs[i].get_xlim()[0], axs[i].get_ylim()[0])
        max_val = max(axs[i].get_xlim()[1], axs[i].get_ylim()[1])
        axs[i].plot([min_val, max_val], [min_val, max_val], alpha=.5, linestyle='--', color='black')
        axs[i].set_title(dataset)


        axs[i].set_xlabel('Predicted')
        if i == 0: axs[i].set_ylabel('Empirical')
        # Decrease size of tick font
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        #else: axs[i].set_yticklabels([])

    bbox_to_anchor = (-1.7,-.2) if is_binary else (-2,-.2)
    plt.legend(fancybox=True, bbox_to_anchor=bbox_to_anchor, ncol=4)

    type = 'binary' if is_binary else 'regression'
    filename = folder + '/calibration_' + type + '.pdf'
    plt.savefig(filename, dpi=1000, bbox_inches="tight")
    #plt.show()
    plt.clf()

# # # CONSISTENCY # # #

def plot_consistency(results, is_binary, algorithms, datasets, folder='figures', idx=42):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))    

    for i, dataset in enumerate(datasets):
        max_depths = sorted(results[dataset][list(algorithms.keys())[0]].keys())
        for j, name in enumerate(algorithms.keys()):
            std= np.array([results[dataset][name][max_depth][idx] for max_depth in max_depths])
            axs[i].plot(max_depths, std, label=name, linestyle=linestyles[j], linewidth=5)
        if i == 0: axs[i].set_ylabel('Uncertainty')

        axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[i].set_xlabel('Depth')
        axs[i].set_title(dataset)

    plt.legend(fancybox=True, bbox_to_anchor=(-1.3,-.2), ncol=4)
    type = 'binary' if is_binary else 'regression'
    filename = folder + '/consistency_' + type + '.pdf'
    plt.savefig(filename, dpi=1000, bbox_inches="tight")
    #plt.show()
    plt.clf()

# # # ABSTENTION # # #

def plot_abstention(results, algorithms, datasets, percentiles, metrics, metric_name='Statistical Parity', folder='figure'):
    fig, axs = plt.subplots(1, len(datasets), figsize=(20, 4))

    for i, dataset in enumerate(datasets):
        instance = results[dataset]['instance']
        for j, algo_name in enumerate(algorithms):
            output = results[dataset][algo_name]
            metric_values = []
            for percentile in percentiles:
                std = output['std']
                include = std <= np.percentile(std, percentile)
                metric_value = metrics[metric_name](
                    output['pred'][include], instance['y_test'][include], instance['group_test'][include] == 1, run_checks=False
                )
                metric_values.append(metric_value)
            # increase line thickness
            axs[i].plot(1-percentiles/100, metric_values, label=algo_name, linestyle=linestyles[j], linewidth=3)
        if i == 0: axs[i].set_ylabel(metric_name)
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        axs[i].set_title(dataset)
        axs[i].set_xlabel('Abstention Rate')

    plt.legend(fancybox=True, bbox_to_anchor=(-1.2,-.2), ncol=4)
    filename = f'{folder}/abstention_{metric_name}.pdf'
    plt.savefig(filename, dpi=1000, bbox_inches="tight")
    #plt.show()
    plt.clf()

# # # REGRESSION FAIRNESS # # #

def plot_regression_fairness(results, datasets, algorithms, folder='figures'):
    fig, axs = plt.subplots(1, len(datasets), figsize=(20, 4))

    for dataset_num, dataset in enumerate(datasets):
        instance = results[dataset]['instance']
        min_z, max_z = min(instance['y_test']), max(instance['y_test'])
        z = np.linspace(min_z, max_z, 1000)

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, algorithm in enumerate(algorithms):
            output = results[dataset][algorithm]
            cdfs = []
            for j, group in enumerate(np.unique(instance['group_test'])):
                cdf = compute_cdf(output, z, instance['group_test'] == group)
                label = algorithm if j == 0 else None
                axs[dataset_num].plot(z, cdf, label=label, color=colors[i], linestyle=linestyles[j])
                cdfs += [cdf]
        axs[dataset_num].tick_params(axis='both', which='major', labelsize=10)
        axs[dataset_num].set_title(dataset)
        axs[dataset_num].set_xlabel(r'$y$')
        if dataset_num == 0:
            axs[dataset_num].set_ylabel('CDF')

    plt.legend(fancybox=True, bbox_to_anchor=(0.8,-.2), ncol=len(algorithms))
    plt.savefig(f'{folder}/regression_fairness.pdf', dpi=1000, bbox_inches='tight')
    #plt.show()
    plt.clf()