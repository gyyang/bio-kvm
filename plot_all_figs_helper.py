from copy import deepcopy
from pprint import pprint
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import joblib
import torch
import scipy.stats as st
from pathlib import Path
from sklearn.linear_model import LinearRegression

from models.model_utils import get_model, MemoryNet, ControllerModel
from datasets.dataset_utils import get_dataset
import configs
from analysis.analysis import evaluate_run
import tools
from training.train import train
import experiments


def _parse_get_model(modeldir_or_model, config=None):
    if type(modeldir_or_model) is str:
        modeldir = modeldir_or_model
        model = None
    else:
        modeldir = None
        model = modeldir_or_model
        assert config is not None, 'Must provide config with model'
    return modeldir, model


def get_acc_vs_seqlen_experiment(experiment, seqlen_list):
    acc_vs_seqlen_per_exp = {'seqlen' : seqlen_list}
    for modeldir in tools.get_modeldirs(experiment):
        acc_vs_seqlen_per_exp[modeldir] = _get_acc_vs_seqlen(modeldir, seqlen_list)
    return acc_vs_seqlen_per_exp


def get_acc_vs_seqlen_configs(configs_dict, seqlen_list=None, repeat=10,
                              fname=None):
    try:
        result = joblib.load(fname)
        seqlen_list = result['seqlen_list']
        acc_vs_seqlen = result['acc_vs_seqlen']
        err_vs_seqlen = result['err_vs_seqlen']
        for label in acc_vs_seqlen:
            assert label in configs_dict
    except:
        if seqlen_list is None:
            seqlen_list = np.unique(np.logspace(0, np.log10(180), 50, dtype=int))
        acc_vs_seqlen = {}
        err_vs_seqlen = {}
        for label, config in configs_dict.items():
            print(label)
            net = get_model(config)
            acc_vs_seqlen[label], err_vs_seqlen[label] = _get_acc_vs_seqlen(
                net, seqlen_list, config=config, repeat=repeat
                )

        if fname is not None:
            result = {
                'seqlen_list':seqlen_list, 'acc_vs_seqlen':acc_vs_seqlen,
                'err_vs_seqlen':err_vs_seqlen
                }
            joblib.dump(result, fname)
    return seqlen_list, acc_vs_seqlen, err_vs_seqlen


def _get_acc_vs_seqlen(modeldir_or_model, seqlen_list, config=None,
                       early_stop_acc=0., repeat='auto'):
    modeldir, model = _parse_get_model(modeldir_or_model, config=config)

    acc_vs_seqlen = np.full(len(seqlen_list), np.nan)
    err_vs_seqlen = [[] for _ in range(len(seqlen_list))]
    for i,seqlen in enumerate(seqlen_list):
        if config is None:
            config = tools.load_config(modeldir)
        config['dataset']['T_min'] = seqlen
        config['dataset']['T_max'] = seqlen

        if repeat == 'auto':
            n_batch = int(max(seqlen_list)/seqlen)
            n_batch = max(10, n_batch) # For error bar evaluations
        else:
            n_batch = repeat
        result = evaluate_run(modeldir=modeldir, model=model, n_batch=n_batch,
                              load_hebb=False, update_config=config, save_pickle=False)
        acc, err = get_avg_recall_acc(result)
        acc_vs_seqlen[i] = acc
        err_vs_seqlen[i] = err

        print('seq len = {}, acc = {} (repeat={})'.format(seqlen, acc, n_batch))
        if acc < early_stop_acc:
            print('Accuracy < {}. Stopping evaluation.\n'.format(early_stop_acc))
            acc_vs_seqlen[i:] = acc
            err_vs_seqlen[i:] = err
            break
    return list(acc_vs_seqlen), list(err_vs_seqlen)


def get_avg_recall_acc(result, recall_idx_list=None):
    """
    Args:
        result (dict): output of evaluate_run
    Returns:
        float: average accuracy for that run, across batches and across target
        float: 95% confidence interval for that run, across batches/target
    """
    acc_per_timepoint_list = result['acc']
    if recall_idx_list is None:
        recall_idx_list = [data['recall_ind'].long() for data in result['data']]
    return _get_avg_recall_acc(acc_per_timepoint_list, recall_idx_list)


def _get_avg_recall_acc(acc_per_timepoint_list, recall_idx_list):
    """
    Args:
        acc_per_timepoint_list (list): list of length num_batches, each entry is a
            length-T tensor, where element t is the accuracy at time t
        recall_idx_list (list): list of length num_batches, each entry is a list
            of indices where the network was asked to recall, i.e. the ones at which
            to compute the accuracy

    Returns:
        float: average accuracy across batches for the indices in recall_idx_list
    """
    assert len(acc_per_timepoint_list) == len(recall_idx_list)

    accs = []
    n_batches = len(acc_per_timepoint_list)
    for acc_per_time, recall_idx in zip(acc_per_timepoint_list, recall_idx_list):
        #loop over batches
        batch_acc = acc_per_time[recall_idx].mean()
        accs.append(batch_acc.item())
    return get_mean_and_err(accs)

def get_mean_and_err(vals):
    sem = st.sem(vals)
    mean = np.mean(vals)
    if (sem == 0) or (len(vals) == 1):
        err = [mean, mean]
    else:
        err = st.t.interval(alpha=0.95, df=len(vals)-1, loc=mean, scale=sem)
    return mean, err

def get_capacity(config, net=None, mode='sequential', thres=0.99, repeat=3):
    if net is None:
        net = get_model(config)
    if mode == 'continual':
        config['dataset']['recall_order'] = 'interleave'
        config['dataset']['p_recall'] = 0.5

    len_lo = 1
    len_hi = 2
    while len_lo < len_hi:
        if mode == 'sequential':
            config['dataset']['T_min'] = len_hi
            config['dataset']['T_max'] = len_hi
        elif mode == 'continual':
            config['dataset']['recall_interleave_delay'] = len_hi
            config['dataset']['T_min'] = config['dataset']['T_max'] = max(1000, len_hi*20)

        result = evaluate_run(model=net, update_config=config,
                              n_batch=repeat, load_hebb=False)
        acc, _ = get_avg_recall_acc(result)
        if acc >= thres:
            len_lo = len_hi
            len_hi = len_hi*2
        else:
            len_hi = int((len_lo+len_hi)/2)
        print('acc={:.2f}, lo={}, hi={}'.format(acc, len_lo, len_hi))
    return len_hi


def plot_capacity(sizes, capacities, labels, label_order, thres, save_fname=None):
    fig, ax = plt.subplots()
    for label in label_order:
        is_label = labels==label
        slope, _, _, _, _ = st.linregress(sizes[labels==label], capacities[labels==label])
        legend_label = label + ' C~{:0.3f}N'.format(slope)
        sns.regplot(x=sizes[is_label], y=capacities[is_label], x_estimator=np.mean,
                    ax=ax, scatter_kws={'s':7.}, line_kws={'label':legend_label, 'linewidth':1.5})
    ax.legend()
    ax.set_xlabel('Network size')
    ax.set_ylabel('Capacity ({}% acc)'.format(int(thres*100)))
    format_and_save(fig, save_fname)
    return ax


def plot_acc_vs_seqlen(seqlen_list, acc_vs_seqlen, err_vs_seqlen, labels=None, save_fname=None):
    fig, ax = plt.subplots()
    legend_handles = []
    if labels is None:
        labels = acc_vs_seqlen.keys()
    for label in labels:
        acc = acc_vs_seqlen[label]
        err = err_vs_seqlen[label]
        handle = ax.plot(seqlen_list, acc, label=label)
        legend_handles.append(handle[0])
        ax.fill_between(seqlen_list, *zip(*err), label=label, alpha=0.3)
    ax.legend(handles=legend_handles)
    ax.set_xlabel('Number of stimuli')
    ax.set_ylabel('Accuracy')
    format_and_save(fig, save_fname)
    return ax


def format_and_save(fig, save_fname=None):
    if save_fname is not None:
        fig.set_size_inches(2.75,1.7)
        fig.tight_layout()
        fig.savefig(save_fname)


# HELPER FUNCTIONS FOR FLASHBULB TASK

def get_flashbulb_performance(
    net_type, n_batch=10, p_ec=0.01,
    vary_ec_strength=False, default_ec_strength=15
    ):

    #R_list = [1,2,3,4,5,7,10,20,30,40,50,60,80,100]
    R_list = np.unique(np.logspace(0, np.log10(180), 30, dtype=int))

    if vary_ec_strength: # Used for Hopfield
        ec_strength_list = [10, 50, 1E3, 1E6] #[10, 50, 1E2, 1E3, 1E4, 1E5]
    else:
        ec_strength_list = [default_ec_strength]

    # Load network configs
    if net_type == "Reference":
        fullconfig = get_reference_flashbulb()
    elif net_type == "Random":
        fullconfig = get_random_reference_flashbulb()
    elif net_type == "Hopfield":
        fullconfig = get_hopfield_flashbulb()
    else:
        raise ValueError("Unknown network")

    # Load dataset configs
    fullconfig['dataset'] = dict()
    fullconfig['dataset']['name'] = 'ecstasy'
    fullconfig['dataset']['stim_dim'] = 40
    fullconfig['dataset']['recall_order'] = 'interleave'
    fullconfig['dataset']['p_recall'] = 0.5
    fullconfig['dataset']['p_ec'] = p_ec
    fullconfig['dataset']['stim_dim'] = 40
    fullconfig['input_size'] = fullconfig['dataset']['stim_dim'] + 1

    net = get_model(fullconfig)

    acc_results = {}
    acc_results['ec_strength_list'] = ec_strength_list
    acc_results['R_list'] = R_list

    fig, ax = plt.subplots()
    for ec_idx, ec_strength in enumerate(ec_strength_list):
        acc_vs_R_reg = [] # For the regular memories
        acc_vs_R_flashbulb = [] # For the special memories
        err_vs_R_reg = []
        err_vs_R_flashbulb = []
        for R in R_list:
            fullconfig['dataset']['recall_interleave_delay'] = R
            fullconfig['dataset']['T_min'] = fullconfig['dataset']['T_max'] = 1000 #max(1000, R*20)
            fullconfig['dataset']['ec_strength'] = ec_strength
            fullconfig['dataset']['num_ec'] = 5
            result = {}
            result['data'] = []
            result['outputs'] = []
            result['acc'] = []
            for _ in range(n_batch):
                _result = evaluate_run(model=net, n_batch=1, load_hebb=False,
                                      update_config=fullconfig, save_pickle=False)
                result['data'].append(_result['data'][0])
                result['outputs'].append(_result['outputs'][0])
                result['acc'].append(_result['acc'][0])

            reg_recall_idx_list = [
                data['nonec_store_recall_map'][1].long() for data in result['data']
                ]
            flashbulb_recall_idx_list = [
                data['ec_store_recall_map'][1].long() for data in result['data']
                ]

            acc_reg, err_reg = get_avg_recall_acc(result, reg_recall_idx_list)
            acc_flashbulb, err_flashbulb = get_avg_recall_acc(
                result, flashbulb_recall_idx_list
                )
            acc_vs_R_reg.append(acc_reg)
            acc_vs_R_flashbulb.append(acc_flashbulb)
            err_vs_R_reg.append(err_reg)
            err_vs_R_flashbulb.append(err_flashbulb)
        acc_results[ec_strength] = {}
        acc_results[ec_strength]['acc_vs_R_reg'] = acc_vs_R_reg
        acc_results[ec_strength]['acc_vs_R_flashbulb'] = acc_vs_R_flashbulb
        acc_results[ec_strength]['err_vs_R_reg'] = err_vs_R_reg
        acc_results[ec_strength]['err_vs_R_flashbulb'] = err_vs_R_flashbulb

    return acc_results

def get_reference_flashbulb():
    fullconfig = deepcopy(configs.get_config(
        'ref_seq', stim_dim=40, hidden_size=40
        ))
    fullconfig['plasticnet']['i2h']['stability'] = True
    fullconfig['plasticnet']['h2o']['stability'] = True
    return fullconfig

def get_random_reference_flashbulb():
    fullconfig = deepcopy(configs.get_config(
        'ref_rand', stim_dim=40, hidden_size=40
        ))
    fullconfig['plasticnet']['i2h']['stability'] = True
    fullconfig['plasticnet']['i2h']['stability_threshold'] = 5.0
    fullconfig['plasticnet']['h2o']['stability'] = True
    return fullconfig

def get_hopfield_flashbulb():
    fullconfig = deepcopy(configs.get_config(
        'hopfield', stim_dim=40
        ))
    fullconfig['hopfield']['steps'] = 3
    fullconfig['hopfield']['decay_rate'] = 0.95
    fullconfig['hopfield']['learning_rate'] = 0.5/40
    return fullconfig


# HELPER FUNCTIONS FOR FIGURE 6


def load_model(exp, filesdir=None):
    if filesdir is None:
        filesdir = Path('files')
    if exp in dir(experiments):
        # Get list of configurations from experiment function
        fullconfig, config_ranges, mode = getattr(experiments,
                                                  exp)()
        fullconfig = deepcopy(fullconfig)
        model_configs = tools.vary_config(fullconfig, config_ranges, mode)
        model_configs = deepcopy(model_configs)
        experiment_found = True
    else:
        experiment_found = False
    if not experiment_found:
        raise ValueError('Model experiment not found: ', exp)
    dataset = get_dataset(model_configs[0]['dataset'], verbose=False)
    input_size = dataset.input_dim
    output_size = dataset.output_dim
    config = deepcopy(model_configs[0])

    if 'ctrlnet' in config.keys():
        model = ControllerModel(
            input_size, output_size, ctrl_config=config['ctrlnet'],
            mem_config=config['plasticnet']
            )
    else:
        model = MemoryNet(input_size, output_size, config=config['plasticnet'])
    try:
        state_dict_path = filesdir / exp / config['model_name'] / 'model.pt'
        model.load_state_dict(torch.load(
            state_dict_path.as_posix(),
            map_location=torch.device('cpu')
            ))
        print(f'Model loaded for {exp}')
    except:
        print(f'Model not loaded for {exp}')
    model.eval()
    return model


def load_dataset(exp, idx=0):
    if exp in dir(experiments):
        # Get list of configurations from experiment function
        fullconfig, config_ranges, mode = getattr(experiments,
                                                  exp)()
        fullconfig = deepcopy(fullconfig)
        model_configs = tools.vary_config(fullconfig, config_ranges, mode)
        experiment_found = True
    else:
        experiment_found = False
    if not experiment_found:
        raise ValueError('Model experiment not found: ', exp)
    dataset = get_dataset(model_configs[idx]['dataset'], verbose=False)
    return dataset


def generate_accs(
    model, dset, dset_param, dset_param_range, num_iters
    ):
    """
    Helper function that returns a list of model accuracies as
    some dataset parameter is changed.
    """

    params = []
    accs = []
    errs = []

    for param_val in dset_param_range:
        _accs = []
        for _ in range(num_iters):
            kwargs = {dset_param: param_val}
            data = dset.generate(**kwargs)
            inputs = torch.from_numpy(data['input']).float()
            target = torch.from_numpy(data['target']).float()
            mask = torch.from_numpy(data['mask']).float()
            if 'modu_input' in data.keys():
                modu_input = torch.from_numpy(data['modu_input']).float()
                modu_input = modu_input.unsqueeze(1)
            else:
                modu_input = None
            inputs = inputs.unsqueeze(1)  # add batch dimension
            outputs, rnn_out = model(inputs, modu_input=modu_input)
            outputs = torch.squeeze(outputs)
            outputs = torch.sign(outputs)
            match = (outputs == target).float()
            mask_sum = torch.sum(mask)
            acc = torch.sum(mask * torch.mean(match, dim=1)) / mask_sum
            _accs.append(acc.item())
        mean, err = get_mean_and_err(_accs)
        params.append(param_val)
        accs.append(mean)
        errs.append(err)

    return params, accs, errs

def get_generalization_curves(
    exps, dset_param, dset_param_range, num_iters=40, filesdir=None
    ):
    """
    For one model, makes plot of accuracy as a function of
    some dataset parameter (sequence length, number of pastes, etc)
    (Fig 6B, 6C, 6D)
    """

    acc_results = {}

    for i, exp in enumerate(exps):
        acc_results[exp] = {}
        model = load_model(exp, filesdir=filesdir)
        dset = load_dataset(exp)
        params, accs, errs = generate_accs(
            model, dset,
            dset_param, dset_param_range,
            num_iters
            )

        acc_results[exp]['params'] = params
        acc_results[exp]['accs'] = accs
        acc_results[exp]['errs'] = errs

    return acc_results


def plot_dset_visualization(
    dset, dset_generate_args=None, show_modu=True
    ):
    """
    For a dataset, plots the visualization of the dataset
    (Fig 6A, 6C)
    """

    if dset_generate_args is not None:
        data = dset.generate(**dset_generate_args)
    else:
        data = dset.generate()

    X = data['input']
    Y = data['target']
    M = data['mask']

    if show_modu:
        modu = data['modu_input']
        X = np.hstack((X, modu))

    pattern_dim = X.shape[1]

    for i, data in enumerate([X, Y]):
        figsize = (3., 2.)
        rect = [0.15, 0.15, 0.65, 0.65]
        rect_cb = [0.82, 0.15, 0.02, 0.65]
        rect_bottom = [0.15, 0.12, 0.65, 0.02]
        rect_left = [0.12, 0.15, 0.02, 0.65]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        im = plt.imshow(data.T, aspect='auto', cmap='RdBu')

        if i == 0:
            title = 'Input X'
            ylabel = 'Stimulus dimension'
        else:
            title = 'Target Y'
            ylabel = 'Target dimension'
        plt.title(title)
        plt.xlabel('Time step', labelpad=15)
        plt.ylabel(ylabel, labelpad=15)
        plt.xticks([])
        plt.yticks([])

        colors = np.array([[55, 126, 184], [228, 26, 28], [178, 178, 178]]) / 255
        labels = M  # From dataset
        texts = ['Store', 'Test']
        tools.add_colorannot(fig, rect_bottom, labels, colors, texts)

        if i == 0:
            # Note: Reverse y axis
            # Innate, flexible
            if show_modu:
                colors = np.array([[245, 110, 128], [149, 0, 149]]) / 255
                labels = np.array([1] + [0]*pattern_dim)
                texts = ['Stimulus', 'Gate']
                tools.add_colorannot(fig, rect_left, labels, colors, texts,
                                     orient='vertical')
            else:
                colors = np.array([[245, 110, 128]]) / 255
                labels = np.array([0]*pattern_dim)
                texts = ['Stimulus']
                tools.add_colorannot(fig, rect_left, labels, colors, texts,
                                     orient='vertical')

        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax)
        cb.outline.set_linewidth(0.5)
        cb.set_label('Weight')
        plt.tick_params(axis='both', which='major')
        plt.show()


def plot_example_output(model, dset, dset_args, random_seed, sign=False):
    """
    For a model and a dataset, plots the model output next to the
    target output (Fig 6D)
    """

    output_size = model.output_size
    np.random.seed(random_seed)
    data = dset.generate(**dset_args)
    X = data['input']
    Y = data['target']
    M = data['mask']

    if 'modu_input' in data.keys():
        modu_input = torch.from_numpy(data['modu_input']).float().to(device)
        modu_input = modu_input.unsqueeze(1)
    else:
        modu_input = None

    inputs = torch.from_numpy(X).float().to(device)
    labels = torch.from_numpy(Y).float().to(device)
    mask = torch.from_numpy(M).bool().to(device)
    inputs = inputs.unsqueeze(1)  # add batch dimension
    outputs, rnn_out = model(inputs, modu_input=modu_input)
    outputs = outputs.view(-1, output_size).detach().numpy()
    fig, axs = plt.subplots(1,2)
    inputs = torch.squeeze(inputs)

    if sign:
        outputs = np.sign(outputs)
    axs[0].imshow(
        outputs[mask,:].T, cmap='RdBu',
        vmin=(outputs[mask,:-1]).min(),
        vmax=(outputs[mask,:-1]).max()
        )
    axs[0].set_title("Output")

    axs[1].imshow(
        Y[mask,:].T, cmap='RdBu'
        )
    axs[1].set_title("Target")

    plt.show()
    return ax
