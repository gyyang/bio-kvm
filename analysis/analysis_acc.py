"""Analyze accuracy."""


import os
import sys
import numpy as np
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
try:
    import seaborn as sns
except:
    pass

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
import configs
import analysis.analysis as analysis


def compute_acc_vs_time(
        baseconfigs,
        time_steps,
        modeldirs=None,
        save_name=None):
    """Compute acc over a list of configs.

    Args:
        baseconfigs: list of configs
        time_steps: list of list of time steps to evaluate each config
        modeldirs: list of modeldirs to load models
        save_name: str, name of saved file
    """
    n_config = len(baseconfigs)  # number of configs
    if modeldirs is None:
        modeldirs = [None for _ in range(n_config)]

    assert n_config == len(time_steps)
    assert n_config == len(modeldirs)

    df = pd.DataFrame()
    # For different stimulus dimensions
    for i in range(n_config):
        print('{}/{} config'.format(i+1, n_config))
        modeldir = modeldirs[i]
        time_step = time_steps[i]
        baseconfig = baseconfigs[i]

        for t in time_step:
            config = deepcopy(baseconfig)
            config['dataset']['T_min'] = t
            config['dataset']['T_max'] = t

            results = analysis.evaluate_run(
                modeldir=modeldir, n_batch=10, load_hebb=False,
                update_config=config, save_pickle=False)

            # Compute accuracy at recall time
            # TODO: This code is specific to the Recall dataset
            tmp_accs = list()
            for tmp_acc, recall_ind in zip(results['acc'],
                                           [data['recall_ind'].long() for data in results['data']]):
                tmp_accs.append(np.mean(tmp_acc[recall_ind].numpy()))
            acc_avg = np.mean(tmp_accs)

            item = {'acc': acc_avg, 't': t}
            item.update(tools.flatten_nested_dict(config))
            df = df.append(item, ignore_index=True)

    if save_name is not None:
        os.makedirs("./files/hopfield", exist_ok=True)
        df.to_pickle("./files/hopfield/" + save_name + "_acc.pkl")
    return df


def _get_baseconfig():
    config = dict()
    config['dataset'] = configs.RecallDatasetConfig
    config['dataset']['p_recall'] = 1.
    return config


def _get_time_range(stim_dim):
    """For a stimulus dimension, get range of time steps."""

    starts = [1] + list(np.array([0.2, 1.0, 3.0]) * stim_dim)
    n_steps = [20, 10, 10]
    time_steps = list()
    for i in range(len(n_steps)):
        start = starts[i]
        stop = starts[i + 1]
        n_step = n_steps[i]
        step = np.ceil((stop - start)/n_step)
        time_steps.append(np.arange(start, stop, step))

    time_steps = np.concatenate(time_steps)
    return time_steps


def compute_acc_vs_stim_dim_special_plastic(path=None):
    if path is None:
        path = './files/compare_hopfield'

    stim_dims = list()
    baseconfigs = list()
    modeldirs = tools.get_modeldirs(path)
    for modeldir in modeldirs:
        config = tools.load_config(modeldir)
        stim_dims.append(config['dataset']['stim_dim'])
        baseconfigs.append(config)
    # Sort by stim dims
    ind_sort = np.argsort(stim_dims)
    stim_dims = [stim_dims[i] for i in ind_sort]
    time_steps = [_get_time_range(d) for d in stim_dims]
    modeldirs = [modeldirs[i] for i in ind_sort]

    results = compute_acc_vs_time(
        baseconfigs=baseconfigs,
        time_steps=time_steps,
        modeldirs=modeldirs,
        save_name='special_plastic')

    return results


def compute_acc_vs_stim_dim_special_plastic_reference():
    stim_dims = np.array([40, 60, 80])
    # stim_dims = np.array([40])

    config = _get_baseconfig()
    config['plasticnet'] = configs.SpecialPlasticRNNReferenceConfig

    baseconfigs = list()
    for stim_dim in stim_dims:
        c = deepcopy(config)
        c['plasticnet']['hidden_size'] = stim_dim
        c['dataset']['stim_dim'] = stim_dim
        baseconfigs.append(c)
    time_steps = [_get_time_range(d) for d in stim_dims]
    results = compute_acc_vs_time(
        baseconfigs=baseconfigs,
        time_steps=time_steps,
        save_name='special_plastic_reference')

    return results


def compute_acc_vs_stim_dim_hopfield():
    stim_dims = np.array([20, 40, 60, 80])

    config = _get_baseconfig()
    config['hopfield'] = configs.HopfieldConfig
    config['hopfield']['steps'] = 5

    baseconfigs = list()
    for stim_dim in stim_dims:
        c = deepcopy(config)
        c['dataset']['stim_dim'] = stim_dim
        baseconfigs.append(c)

    time_steps = [_get_time_range(d) for d in stim_dims]
    results = compute_acc_vs_time(
        baseconfigs=baseconfigs,
        time_steps=time_steps,
        save_name='hopfield')

    return results


def compute_acc_vs_stim_dim_special_plastic_ref_randomtf(
        local_thirdfactor_prob=0.1):
    if hasattr(local_thirdfactor_prob, '__iter__'):
        results = list()
        for p in local_thirdfactor_prob:
            r = compute_acc_vs_stim_dim_special_plastic_ref_randomtf(p)
            results.append(r)
        return results

    stim_dims = np.array([60])

    config = _get_baseconfig()
    config['plasticnet'] = configs.SpecialPlasticRNNReferenceConfig
    config['plasticnet']['local_thirdfactor_mode'] = 'random'
    config['plasticnet']['local_thirdfactor_prob'] = local_thirdfactor_prob

    baseconfigs = list()
    for stim_dim in stim_dims:
        c = deepcopy(config)
        c['plasticnet']['hidden_size'] = stim_dim
        c['dataset']['stim_dim'] = stim_dim
        baseconfigs.append(c)
    time_steps = [_get_time_range(d) for d in stim_dims]
    results = compute_acc_vs_time(
        baseconfigs=baseconfigs,
        time_steps=time_steps,
        save_name='spnref_randomtf_{:0.2f}'.format(local_thirdfactor_prob))

    return results


def plot_acc_vs_stim_dim_singlemodel(save_name):
    df = pd.read_pickle("./files/hopfield/" + save_name + "_acc.pkl")
    df_stimdim = df['dataset.stim_dim']
    stim_dims = df_stimdim.unique()

    # Compute capacity
    acc_th = 0.99
    capacity = list()
    for stim_dim in stim_dims:
        ts = df['t'][df_stimdim == stim_dim]
        accs = df['acc'][df_stimdim == stim_dim]
        ind = np.where(accs > acc_th)[0][-1]
        capacity.append(ts.to_numpy()[ind])
    capacity = np.array(capacity)

    plt.figure(figsize=(3, 2))
    for stim_dim in stim_dims:
        ts = df['t'][df_stimdim == stim_dim]
        accs = df['acc'][df_stimdim == stim_dim]
        plt.plot(ts, accs, label=str(stim_dim))
    plt.legend(title='Network size')
    plt.xlabel('Seq len')
    plt.ylabel('Accuracy')

    for xlim_max in [None, np.max(capacity/stim_dims) * 1.5]:
        plt.figure(figsize=(3, 2))
        for stim_dim in stim_dims:
            ts = df['t'][df_stimdim == stim_dim]
            accs = df['acc'][df_stimdim == stim_dim]
            plt.plot(ts / stim_dim, accs, label=str(stim_dim))
        plt.legend(title='Network size')
        plt.xlabel('Seq len / Network size')
        plt.ylabel('Accuracy')
        plt.ylim([0.7, 1.05])
        if xlim_max is not None:
            plt.xlim([0, xlim_max])
        plt.grid()

    def _fit(x, y):
        x_fit = np.linspace(x[0], x[-1], 3)
        model = LinearRegression()
        model.fit(x[:, np.newaxis], y)
        y_fit = model.predict(x_fit[:, np.newaxis])
        return x_fit, y_fit, model

    plt.figure(figsize=(3, 3))
    plt.plot(stim_dims, capacity, 'o-')
    plt.xlabel('Network size')
    plt.ylabel('Capacity')

    x_fit, y_fit, model = _fit(stim_dims, capacity)
    label = r'$y ={:0.2f}x + {:0.2f}$'.format(model.coef_[0], model.intercept_)
    plt.plot(x_fit, y_fit, label=label)
    plt.legend()
    plt.title('Capacity empirically measured at acc threshold {:0.2f}'.format(
        acc_th),
              fontsize=7)


def plot_acc_vs_stim_dim_multimodel(save_names):
    plt.figure(figsize=(3, 2))
    for i, save_name in enumerate(save_names):
        color = sns.color_palette()[i]
        df = pd.read_pickle("./files/hopfield/" + save_name + "_acc.pkl")
        df_stimdim = df['dataset.stim_dim']
        stim_dims = df_stimdim.unique()
        for j, stim_dim in enumerate(stim_dims):
            ts = df['t'][df_stimdim == stim_dim]
            accs = df['acc'][df_stimdim == stim_dim]
            if j == 0:
                plt.plot(ts / stim_dim, accs, label=save_name, color=color,
                         alpha=0.5)
            else:
                plt.plot(ts / stim_dim, accs, color=color, alpha=0.5)
    plt.legend(title='Network type')
    plt.xlabel('Seq len / Network size')
    plt.ylabel('Accuracy')
    plt.ylim([0.7, 1.05])
    plt.grid()
    tools.save_fig('compare_hopfield', '.'.join(save_names))


def plot_acc_vs_stim_dim(save_name):
    if isinstance(save_name, str):
        plot_acc_vs_stim_dim_singlemodel(save_name)
    elif isinstance(save_name, list):
        plot_acc_vs_stim_dim_multimodel(save_names=save_name)
    else:
        raise ValueError('Unknown save name', save_name)


if __name__ == '__main__':
    pass
    # plot_acc_vs_stim_dim()
