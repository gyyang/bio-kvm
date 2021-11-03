"""Analysis file testing whether the network can have very long-term memory."""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from analysis.analysis import get_errorbar
from analysis.analysis import evaluate_run


def evaluate_longrun(save_path, T=1000):
    # Evaluate network post training
    T_total = 10000
    n_batch = max((1, int(T_total/T)))
    update_config = {'dataset': {'T_min': T, 'T_max': T}}
    evaluate_run(modeldir=save_path, n_batch=n_batch,
                 update_config=update_config, save_pickle=True)


def evaluate_ecstasy_longrun(save_path, T=1000):
    """Evaluate network post training on ecstasy."""
    T_total = 50000
    n_batch = max((1, int(T_total/T)))
    dataset_config = {'name': 'ecstasy', 'T_min': T, 'T_max': T, 'p_ec': 1e-2}
    update_config = {'dataset': dataset_config}
    evaluate_run(modeldir=save_path, n_batch=n_batch, update_config=update_config,
                 fname='evaluate_ecstasy_run', save_pickle=True)


def _get_interval_acc(acc, store_recall_map, logx=True):
    """Helper function to get accuracy given memory intervals

    :param acc: list of n_batch acc arrays.
    :param store_recall_map: list of store_recall_maps

    Returns:
        bins_center
        acc_mean
        acc_err
    """
    interval = list()
    new_acc = list()
    for i, acc_batch in enumerate(acc):  # go through batches
        # acc at recall
        new_acc.append(acc_batch[store_recall_map[i][1]])
        interval.append(store_recall_map[i][1] - store_recall_map[i][0])

    acc = np.concatenate(new_acc)
    interval = np.concatenate(interval)

    indsort = np.argsort(interval)
    acc = acc[indsort]
    interval = interval[indsort]

    if logx:
        interval = np.log10(interval)
        v_min, v_max = np.min(interval), np.max(interval)
        # v_min, v_max = np.percentile(interval, [1, 99])
        bins = np.linspace(np.round(v_min, decimals=1),
                           np.round(v_max, decimals=1), 20)
    else:
        bins = np.linspace(-0.5, np.max(interval)+4.5, 20)
    bins_center = (bins[:-1] + bins[1:]) / 2
    indbins = np.searchsorted(interval, bins)

    acc_binned = [acc[indbins[j]: indbins[j + 1]] for j in range(len(indbins) - 1)]
    acc_mean, acc_err = get_errorbar(acc_binned)

    return bins_center, acc_mean, acc_err


def plot_interval_acc(save_path, fname=None, logx=True):
    """Plot accuracy as a function of interval."""
    if fname is None:
        fname = 'evaluate_run.pkl'
    elif fname[-4:] != '.pkl':
        fname = fname + '.pkl'

    with open(os.path.join(save_path, fname), 'rb') as f:
        results = pickle.load(f)

    has_ec = 'ec_store_recall_map' in results

    bins_center, acc_mean, acc_err = _get_interval_acc(
        results['acc'], results['store_recall_map'], logx=logx)

    chance = results['chance'][0]

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    plt.errorbar(bins_center, acc_mean, yerr=acc_err, label='regular')
    plt.plot(bins_center[[0, -1]], [chance] * 2, '--', color='black')

    if has_ec:
        ec_bins_center, ec_acc_mean, ec_acc_err = _get_interval_acc(
            results['acc'], results['ec_store_recall_map'], logx=logx)
        plt.errorbar(ec_bins_center, ec_acc_mean, yerr=ec_acc_err,
                     label='ecstatic')
        plt.legend()

    if logx:
        xlabel = 'Interval (log10)'
        # plt.xticks(ticks=bins_center, labels=[str(int(10**b)) for b in bins_center])
    else:
        xlabel = 'Interval'
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy')
    plt.ylim(top=1.05, bottom=chance-0.1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    tools.save_fig(save_path, fname[:-4] + '_acc_vs_interval')
