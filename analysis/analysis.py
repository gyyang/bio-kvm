"""Analyze the trained models."""

import sys
import os
import pickle
import warnings
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import torch
from torch.utils.tensorboard import SummaryWriter

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename, save_fig
from datasets.dataset_utils import get_dataset
from models.model_utils import get_model


mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['figure.facecolor'] = 'None'

figpath = os.path.join(rootpath, 'figures')


def _get_ax_args(xkey, ykey):
    rect = (0.3, 0.35, 0.5, 0.55)
    ax_args = {}
    return rect, ax_args


def plot_progress(save_path, select=None, exclude=None,
                  legend_key=None, x_range=None, ykeys=None, ax_args=None):
    """Plot progress through training."""

    def _plot_progress(xkey, ykey, modeldirs):
        """Plot progress for one xkey and ykey pair."""
        if ax_args is None:
            rect, ax_args_ = _get_ax_args(xkey, ykey)
        else:
            rect = [0.25, 0.3, 0.6, 0.5]
            ax_args_ = ax_args

        n_model = len(modeldirs)

        logs = [tools.load_log(d) for d in modeldirs]
        cfgs = [tools.load_config(d) for d in modeldirs]

        figsize = (3.5, 2)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect, **ax_args_)

        colors = [cm.cool(x) for x in np.linspace(0, 1, n_model)]

        for i in range(n_model):
            x, y, c = logs[i][xkey], logs[i][ykey], colors[i]
            if x_range:
                x, y = x[x_range[0]: x_range[1]], y[x_range[0]: x_range[1]]
            ax.plot(x, y, color=c, linewidth=1)
            ax.text(x[-1]*1.05, y[-1], nicename(y[-1], mode=ykey), color=c)

        if legend_key is not None:
            # Check if legend_key is string or tuple
            legend_key_ = [legend_key] if isinstance(legend_key, str) else legend_key
            legends = []
            # Loop over curves/mdoels
            for i in range(n_model):
                cfg = tools.flatten_nested_dict(cfgs[i])
                l_list = []
                # Loop over possible tuple of legend keys
                for lk in legend_key_:
                    # TODO: make more general
                    if lk in cfg:
                        nn = nicename(cfg[lk], mode=lk)
                    elif lk == 'plasticnet': #this is a hack
                        lk = 'plasticnet.network'
                        nn = nicename(cfg[i], mode=lk)
                    elif any([k.startswith(lk) for k in cfg]):
                        nn = ' '.join([nicename(cfg[i], mode=k) for k in cfg
                                       if k.startswith(lk)])
                    else: #should not execute, but won't break if it does, just makes ugly figure
                        warnings.warn('Key {} not found in log from {}'.format(lk, save_path))
                        nn = 'Unknown key: {}'.format(lk)
                    l_list.append(nn)
                legends.append(', '.join(l_list))
            title = ', '.join([nicename(lk) for lk in legend_key_])

            ax.legend(legends, fontsize=7, frameon=False, ncol=2, loc='best')
            plt.title(title, fontsize=7)

        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if x_range:
            ax.set_xlim([x_range[0], x_range[1]])
        else:
            ax.set_xlim([-1, logs[0][xkey][-1]])

        figname = '_' + ykey
        if select:
            for k, v in select.items():
                figname += k + '_' + str(v) + '_'

        if x_range:
            figname += '_x_range_' + str(x_range[1])
        save_fig(save_path, figname)

    modeldirs = tools.get_modeldirs(save_path, select_dict=select,
                                    exclude_dict=exclude)
    if not modeldirs:
        print('No model to plot progress')
        return

    if ykeys is None:
        ykeys = []

    if ykeys == 'all':
        log = tools.load_log(modeldirs[0])
        ykeys = [k for k, v in log.items() if v.shape == log['steps'].shape]

    if isinstance(ykeys, str):
        ykeys = [ykeys]

    for plot_var in ykeys:
        _plot_progress('steps', plot_var, modeldirs)


def get_errorbar(x):
    """Get errorbar.

    Args:
        x: list of lists.

    Returns:
        x_mean: list of mean values
        x_err: array (2, N) lower error and upper error for 95% conf interval
    """
    x_mean = [np.mean(x0) for x0 in x]
    x_err = np.zeros((2, len(x_mean)))
    for i, a in enumerate(x):
        bootstrap = [np.mean(np.random.choice(a, size=len(a))) for _ in
                     range(100)]
        x_err[0, i] = x_mean[i] - np.percentile(bootstrap, 2.5)
        x_err[1, i] = np.percentile(bootstrap, 97.5) - x_mean[i]
    return x_mean, x_err


def plot_results(path, xkey, ykey, loop_key=None, select=None,
                 logx=None, logy=False, figsize=None, ax_args=None,
                 plot_args=None, ax_box=None, res=None, string='',
                 plot_actual_value=True):
    """Plot results for varying parameters experiments.

    Args:
        path: str, model save path
        xkey: str, key for the x-axis variable
        ykey: str, key for the y-axis variable
        loop_key: str, key for the value to loop around
        select: dict, dict of parameters to select
        logx: bool, if True, use log x-axis
        logy: bool, if True, use log x-axis
    """
    if isinstance(ykey, str):
        ykeys = [ykey]
    else:
        ykeys = ykey

    if res is None:
        res = tools.load_results(path, select=select)

    tmp = res[xkey][0]
    xkey_is_string = isinstance(tmp, str) or tmp is None

    if plot_args is None:
        plot_args = {}

    # Unique sorted xkey values
    xvals = sorted(set(res[xkey]))

    if logx is None:
        logx = False

    if figsize is None:
        if xkey == 'lr':
            figsize = (4.5, 1.5)
        else:
            figsize = (1.5, 1.2)

    def _plot_results(ykey):
        # Default ax_args and other values, based on x and y keys
        rect, ax_args_ = _get_ax_args(xkey, ykey)
        if ax_args:
            ax_args_.update(ax_args)
        if ax_box is not None:
            rect = ax_box

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect, **ax_args_)
        if loop_key:
            loop_vals = np.unique(res[loop_key])
            colors = [cm.cool(x) for x in np.linspace(0, 1, len(loop_vals))]
            for loop_val, color in zip(loop_vals, colors):
                ind = res[loop_key] == loop_val
                x_plot = res[xkey][ind]
                y_plot = res[ykey][ind]
                if logx:
                    x_plot = np.log(x_plot)
                if logy:
                    y_plot = np.log(y_plot)
                # x_plot = [str(x).rsplit('/', 1)[-1] for x in x_plot]
                ax.plot(x_plot, y_plot, 'o-', markersize=3, color=color,
                        label=nicename(loop_val, mode=loop_key), **plot_args)
        else:
            # Organize
            yvals = list()
            yvals_all = list()
            for xval in xvals:
                yval_tmp = [res[ykey][i] for i, r in enumerate(res[xkey]) if
                            r == xval]
                yval_tmp = np.array(yval_tmp)
                yval_tmp = yval_tmp.flatten()
                if logy:
                    yval_tmp = np.log(yval_tmp)

                yvals.append(np.mean(yval_tmp))
                yvals_all.append(yval_tmp)
            y_mean, y_error = get_errorbar(yvals_all)

            if xkey_is_string:
                x_plot = np.arange(len(xvals))
            else:
                if logx:
                    x_plot = np.log(np.array(xvals))
                else:
                    x_plot = xvals
            # ax.plot(x_plot, y_mean, fmt='o-', markersize=3, **plot_args)
            ax.errorbar(x_plot, y_mean, yerr=y_error, fmt='o-', markersize=3,
                        **plot_args)

            if plot_actual_value:
                for x, y in zip(x_plot, y_mean):
                    if y > ax.get_ylim()[-1]:
                        continue

                    ytext = '{:0.2f}'.format(y)
                    ax.text(x, y, ytext, fontsize=6,
                            horizontalalignment='center',
                            verticalalignment='bottom')

        if 'xticks' in ax_args_.keys():
            xticks = ax_args_['xticks']
            ax.set_xticks(xticks)
        else:
            xticks = x_plot
            xticklabels = [nicename(x, mode=xkey) for x in xvals]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

        # ax.set_xticks(xticks)
        # if not xkey_is_string:
        #     x_span = xticks[-1] - xticks[0]
        #     ax.set_xlim([xticks[0]-x_span*0.05, xticks[-1]+x_span*0.05])
        # ax.set_xticklabels(xticklabels)

        if 'yticks' in ax_args_.keys():
            yticks = ax_args_['yticks']
            if logy:
                ax.set_yticks(np.log(yticks))
                ax.set_yticklabels(yticks)
            else:
                ax.set_yticks(yticks)
        else:
            plt.locator_params(axis='y', nbins=3)

        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))

        if loop_key:
            l = ax.legend(loc=1, bbox_to_anchor=(1.0, 0.5), fontsize= 7,
                          frameon=False, ncol=2)
            l.set_title(nicename(loop_key))

        figname = '_' + ykey + '_vs_' + xkey
        if loop_key:
            figname += '_vary' + loop_key
        if select:
            for k, v in select.items():
                if isinstance(v, list):
                    v = [x.rsplit('/',1)[-1] for x in v]
                    v = str('__'.join(v))
                else:
                    v = str(v)
                figname += k + '_' + v + '__'
        figname += string

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        tools.save_fig(path, figname)

    for ykey in ykeys:
        _plot_results(ykey)


def plot_value_in_time(X, name, save_path=None):
    """Plot X with imshow."""
    fig = plt.figure(figsize=(3, 2.5))
    plt.imshow(X.T, aspect='auto')
    plt.title(name)
    plt.xlabel('Position')
    plt.ylabel(name)
    plt.colorbar()
    if save_path is not None:
        save_fig(save_path, 'analyze_'+name)


def plot_matrix_in_time(X, name, save_path=None, n_plot=5):
    """Plot X at each time point

    args:
        X: (n_time, .., ..)
    """
    n_time = X.shape[0]
    n_time_plot = np.linspace(0, n_time - 1, n_plot, dtype=int)
    vlim = np.max(np.abs(X))
    fig, axes = plt.subplots(1, n_plot, sharey=True, figsize=(7, 1.5))
    for i, ind in enumerate(n_time_plot):
        ax = axes[i]
        im = ax.imshow(X[ind], aspect='auto', vmin=-vlim, vmax=vlim)
        ax.set_title('Time {:d}'.format(ind))
    # f.colorbar(im, ax=ax)
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(name, y=1.08)
    if save_path is not None:
        save_fig(save_path, 'analyze_' + name)


def evaluate_run(modeldir=None, update_config=None, model=None, custom_data=None,
                 n_batch=1, analyze=False, load=True, load_hebb=True,
                 reset_hebb=True, fname=None, save_pickle=False, save_path=None):
    """Evaluate the network for batches of data post training.

    Args:
        modeldir: if None, then do not load models
        update_config: optional update of config loaded from save_path
        model: if not None, use this model and disregard modeldir and
            update_config
        custom_data: optional. Output of dataset.generate(), for evaluating net
            on a particular instance of the dataset (must have n_batch=1)
        n_batch: number of dataset batch to run
        analyze: if True, network in analyze mode
        load_hebb: if True, load Hebbian weights
        reset_hebb: if True, reset Hebbian weights for each batch
        fname: str, filename to store
        save_pickle: if True, save results to pickle file
        save_path: path to save results
    """
    if modeldir is None:
        config = dict()
    else:
        config = tools.load_config(modeldir)
    if update_config is not None:
        config = tools.nested_update(config, update_config)

    # Training networks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if custom_data is None:
        dataset = get_dataset(config['dataset'], verbose=False)
        output_size = dataset.output_dim
    else:
        assert n_batch==1, "Only one batch of custom_data can be passed in"
        seq_len, output_size = custom_data['target'].shape

    if model is None:
        net = get_model(config)
        if load and 'save_path' in config:
            model_path = os.path.join(config['save_path'], 'model.pt')
            net.load(model_path, map_location=device, load_hebb=load_hebb)
    else:
        net = model

    if analyze:
        net.analyze()

    torch.no_grad()

    # TODO: Consider a different way to store
    results = defaultdict(list)
    for batch in range(n_batch):
        if reset_hebb:
            for m in net.modules():
                if 'reset_hebb' in dir(m):
                    m.reset_hebb()

        if custom_data is None:
            data = dataset.generate()
        else:
            data = custom_data
        results['data'].append(data)

        for key, val in data.items():
            data[key] = torch.from_numpy(val).float().to(device)

        # Add batch dimension
        data['input'] = data['input'].unsqueeze(1)
        if 'modu_input' in data.keys():
            data['modu_input'] = data['modu_input'].unsqueeze(1)

        if 'modu_input' in data.keys():
            # TODO: This forces all networks to accept modu_input, fix?
            if 'input_heteroassociative' in data.keys():
                target = data['input_heteroassociative']
            else:
                target = None
            outputs, rnn_out = net(input=data['input'],
                                  modu_input=data['modu_input'], 
                                  target=target)
        else:
            outputs, rnn_out = net(input=data['input'])

        outputs = outputs.view(-1, output_size)
        outputs = torch.sign(outputs)
        results['outputs'].append(outputs)

        # Get acc at recall times
        match = (outputs == data['target']).float()
        acc = match.mean(dim=1)
        results['acc'].append(acc)

    if analyze:
        results.update(net.writer_dict())

    results['config'] = config

    if save_pickle:
        if fname is None:
            fname = 'evaluate_run.pkl'
        elif fname[-4:] != '.pkl':
            fname = fname + '.pkl'

        if save_path is None:
            if modeldir is not None:
                save_path = modeldir
            else:
                raise ValueError('No save_path or modeldir provided')

        with open(os.path.join(save_path, fname), 'wb') as f:
            pickle.dump(results, f)

    return dict(results) #don't want defaultdict, may hide bugs downstream


if __name__ == '__main__':
    pass
