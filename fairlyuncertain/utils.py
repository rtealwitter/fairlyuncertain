import scipy
import numpy as np
import warnings

# # # FORMATTING # # #

def fancy_round(x, precision=3):
    return float(np.format_float_positional(x, precision=precision, unique=False, fractional=False, trim='k'))

def print_table(results_all, filename, include_var=True, lower_is_better=True):
    with open(filename, 'w') as f:
        col_types = 'l' + 'c' * len(results_all.keys())

        f.write('\\begin{tabular} {' + col_types + '}\n')
        f.write('\\toprule\n')

        col_names = ['Approach'] + list(results_all.keys())
        col_names_bold = ['\\textbf{' + col_name + '}' for col_name in col_names]

        f.write(' & '.join(col_names_bold) + ' \\\\ \\midrule\n')

        current_round = lambda x: round(fancy_round(x, 3), 3)
        
        mean_cols = {}
        for dataset in results_all:
            mean_col = []
            for algo_name in results_all[dataset]:
                mean_col.append(current_round(np.mean(results_all[dataset][algo_name])))
            if 'Included' in dataset:
                lower_is_better = not lower_is_better
            mean_col_sorted = sorted(mean_col, reverse=not lower_is_better)
            mean_cols[dataset] = mean_col_sorted

        first_key = list(results_all.keys())[0]

        for algo_name in results_all[first_key]:
            to_print = [algo_name]
            for dataset in results_all:
                mean_val = current_round(np.mean(results_all[dataset][algo_name]))
                std_val = current_round(np.std(results_all[dataset][algo_name]))
                color = ''
                if mean_val == mean_cols[dataset][0]:
                    color = '\\cellcolor{gold!30}'
                elif mean_val == mean_cols[dataset][1]:
                    color = '\\cellcolor{silver!30}'
                elif mean_val == mean_cols[dataset][2]:
                    color = '\\cellcolor{bronze!30}'              
                mean_val_str = f'{color}{mean_val}'
                
                if include_var:
                    to_print.append(f'{mean_val_str} $\\pm$ {std_val}')
                else:
                    to_print.append(mean_val_str)

            f.write(' & '.join(to_print) + ' \\\\ \n')

        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')

def get_consistency_table(results, datasets, algorithms):
    table = {}
    for dataset in datasets:
        table[dataset] = {algo_name : [] for algo_name in algorithms}
        for algo_name in algorithms.keys():            
            max_depths = sorted(results[dataset][algo_name].keys())
            for i in range(len(results[dataset][algo_name][max_depths[0]])):
                std = np.array([results[dataset][algo_name][max_depth][i] for max_depth in max_depths])
                max_std = (np.std(std, axis=0)).max()
                table[dataset][algo_name].append(max_std)
    
    return table

def bold_maxes(rows):
    names = rows[:,0]
    X = np.array(rows[:,1:], dtype=float)
    col_sorted = []
    for col in X.T:
        col_sorted.append(sorted(col))

    for j, row in enumerate(X):
        to_print = [names[j]]
        for i, value in enumerate(row):
            rounded = round(value, 3)
            val = str(rounded)
            if value == col_sorted[i][0]:
                val = f'\\textbf{{{rounded}}}'
            elif value == col_sorted[i][1]:
                val = '\\textbf{\\textit{' + str(rounded) + '}}'
            elif value == col_sorted[i][2]:
                val = '\\textbf{\\underline{' + str(rounded) + '}}'
            to_print.append(val)
        print(' & '.join(to_print) + ' \\\\ \\hline')

# # # CALIBRATION # # # 

def get_binary_nll(pred, std, y):
    std = np.clip(std, 0, .5)
    ptilde_a = (1 + np.sqrt(1 - 4 * std**2))/2
    ptilde_b = (1 - np.sqrt(1 - 4 * std**2))/2
    ptilde = (pred > .5) * ptilde_a + (pred <= .5) * ptilde_b
    ptilde = np.clip(ptilde, 0.001, .999)
    return -1 * (np.log(ptilde) * y + np.log(1-ptilde) * (1-y))

def get_regression_nll(pred, std, y):
    return np.log(std * np.sqrt(2 * np.pi)) + (y - pred)**2 / (2 * std**2)

def get_nll(pred, std, y, is_binary):
    if is_binary:
        return get_binary_nll(pred, std, y)
    else:
        return get_regression_nll(pred, std, y)
    
# # # BINARY FAIRNESS # # #
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def get_diffs(lst):
    max_diff = 0
    for i in range(len(lst)):
        for j in range(len(lst)):
            diff = np.abs(lst[i] - lst[j])
            max_diff = max(max_diff, diff)
    return max_diff

def binary_statistical_parity(pred, y, group, run_checks=True):
    mean_preds = [] 
    for g in np.unique(group):
        include = group == g
        mean_pred = pred[include].mean()
        mean_preds.append(mean_pred)
    val1 = get_diffs(mean_preds)

    if not run_checks:
        return val1
    print('running checks')
    
    val2 = demographic_parity_difference(y, pred, sensitive_features=group)

    error_msg = 'Error in SP...' + str(val1) + '!=' + str(val2)
    assert val1 == val2, error_msg

    return val1


def binary_equalized_odds(pred, y, group, run_checks=True):
    # True positive rate difference
    # False positive rate difference
    # Return the maximum of the two
    tprs, fprs = [], []
    for g in np.unique(group):
        include = group == g
        tpr = np.mean(pred[include][y[include] == 1])
        fpr = np.mean(pred[include][y[include] == 0])
        if np.isnan(tpr): tpr = 0
        if np.isnan(fpr): fpr = 0
        tprs += [tpr]
        fprs += [fpr]
    tpr_diff = get_diffs(tprs)
    fpr_diff = get_diffs(fprs)

    val1 = max(tpr_diff, fpr_diff)

    if not run_checks:
        return val1
    
    print('running checks')

    val2 = equalized_odds_difference(y, pred, sensitive_features=group)

    error_msg = 'Error in EO...' + str(val1) + '!=' + str(val2)

    assert val1 == val2, error_msg

    return val1

def binary_error_rate(pred, y, group, run_checks=True):
    return np.mean(pred != y)
    
binary_metrics = {
    'Error Rate': binary_error_rate,
    'Statistical Parity': binary_statistical_parity, 
    'Equalized Odds': binary_equalized_odds,
    #'Equal Opportunity': binary_equal_opportunity
}

# # # REGRESSION FAIRNESS # # #

def compute_cdf(output, z, group_indicator=None):
    z = z.reshape(-1, 1)
    if group_indicator is None:
        group_indicator = np.ones(len(output['pred']), dtype=bool)

    mus = output['pred'][group_indicator].squeeze()
    if 'std' in output:
        sigma = output['std'][group_indicator]
        cdf = scipy.stats.norm.cdf(z, mus, sigma).mean(axis=1)
    else:
        cdf = (mus < z).mean(axis=1)
    return cdf

def general_sp(cdfs):
    max_diff = 0
    for i in range(cdfs.shape[0]):
        for j in range(cdfs.shape[0]):
            if i == j:
                continue
            diff = np.abs(cdfs[i] - cdfs[j]).max()
            max_diff = max(max_diff, diff)
    return max_diff

def get_regression_statistical_parity(y, group, output):
    z = np.linspace(min(y), max(y), 1000)
    cdfs = []
    for g in np.unique(group):
        cdf = compute_cdf(output, z, group == g)
        cdfs += [cdf]
    return general_sp(np.array(cdfs))
