"""Analyze special plastic network."""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import configs
import analysis.analysis as analysis


def plot_norm_change(modeldir=None, config=None, model=None):
    if config is None:
        config = dict()
        config['dataset'] = configs.RecallDatasetConfig
        config['dataset']['p_recall'] = 1.
        config['dataset']['T_min'] = 1000
        config['dataset']['T_max'] = 1000
        config['plasticnet'] = configs.SpecialPlasticRNNReferenceConfig
        config['plasticnet']['local_thirdfactor_mode'] = 'random'
        config['plasticnet']['local_thirdfactor_prob'] = 0.1

    results = analysis.evaluate_run(
        modeldir=modeldir, update_config=config, model=model,
        n_batch=1, load_hebb=False, analyze=True,
        save_pickle=False)

    def _plot_norm_change(key):
        data = results[key]
        n_time = data.shape[0]

        # Compute norm in time
        norm = np.array([np.linalg.norm(data[i]) for i in range(n_time)])
        diff_norm = np.array(
            [np.linalg.norm(data[i + 1] - data[i]) for i in range(n_time - 1)])

        fig, axs = plt.subplots(2, 1, sharex='all')
        ax = axs[0]
        ax.set_title('Norm of ' + key)
        ax.plot(norm)
        ax = axs[1]
        ax.set_title('Norm of change in ' + key)
        ax.plot(diff_norm)

    for key in ['rnn.i2h.hebb', 'rnn.h2o.hebb']:
        _plot_norm_change(key)


def plot_prob_memory_not_overwritten():
    """Plot probability of memory not overwritten for random third factors."""
    N = 100  # Number of neurons
    p = np.linspace(0, 0.3, 1000)  # probability of third factors
    plt.figure(figsize=(3, 2))
    for T in [50, 75, 100]:  # sequence length
        plt.plot(p, 1 - (1 - (1 - p) ** T) ** (N * p), label=str(T))
    plt.xlabel('Local random third factor prob.')
    plt.ylabel('Prob. of memory not overwritten')
    plt.legend(title='T')
    plt.title('N={:d}'.format(N))
