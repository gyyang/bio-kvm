"""General tools.

Make sure it doesn't import from any other files in this project.
"""

import os
import json
import pickle
from copy import deepcopy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


rootpath = os.path.dirname(os.path.abspath(__file__))
FIGPATH = os.path.join(rootpath, 'figures')
FILEPATH = os.path.join(rootpath, 'files')

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'


def save_fig(path, name, dpi=300, pdf=False, show=False):
    save_name = path.split('files/')[-1]  # hack
    figpath = os.path.join(FIGPATH, save_name)
    os.makedirs(figpath, exist_ok=True)
    figname = os.path.join(figpath, name)
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)
    print('Figure saved at: ' + figname)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True, dpi=dpi)
    if show:
        plt.show()
    # plt.close()


def save_config(config, save_path, also_save_as_text=True):
    """Save config."""
    # config_dict = config.todict()
    config_dict = config
    # print(config_dict)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)

    if also_save_as_text:
        with open(os.path.join(save_path, 'config.txt'), "w") as f:
            for k, v in config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')


def load_config(save_path):
    """Load config."""
    try:
        with open(os.path.join(save_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
    except:
        import pickle
        with open(os.path.join(save_path, 'config.p'), 'rb') as f:
            config_dict = pickle.load(f)
    return config_dict


def vary_config(base_config, config_ranges, mode):
    """Return configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
        mode: str, can take 'combinatorial', 'sequential', and 'control'

    Return:
        configs: a list of config dict [config1, config2, ...]
    """
    if mode == 'combinatorial':
        _vary_config = _vary_config_combinatorial
    elif mode == 'sequential':
        _vary_config = _vary_config_sequential
    elif mode == 'control':
        _vary_config = _vary_config_control
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))
    configs, config_diffs = _vary_config(base_config, config_ranges)
    # Automatic set names for configs
    # configs = autoname(configs, config_diffs)
    for i, config in enumerate(configs):
        config['model_name'] = str(i).zfill(6)  # default name
    return configs


def _vary_config_combinatorial(base_config, config_ranges):
    """Return combinatorial configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all possible combinations of hp1, hp2, ...
        config_diffs: a list of config diff from base_config
    """
    # Unravel the input index
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.prod(dims))

    configs, config_diffs = list(), list()
    for i in range(n_max):
        new_config = deepcopy(base_config)

        config_diff = dict()
        indices = np.unravel_index(i, dims=dims)
        # Set up new config
        for key, index in zip(keys, indices):
            new_val = config_ranges[key][index]
            if '.' in key:
                nested_set(new_config, key.split('.'), new_val)
            else:
                new_config[key] = new_val
            config_diff[key] = new_val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _vary_config_sequential(base_config, config_ranges):
    """Return sequential configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 together sequentially
        config_diffs: a list of config diff from base_config
    """
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = dims[0]

    configs, config_diffs = list(), list()
    for i in range(n_max):
        new_config = deepcopy(base_config)
        config_diff = dict()
        for key in keys:
            # setattr(config, key, hp_ranges[key][i])
            new_val = config_ranges[key][i]
            if '.' in key:
                nested_set(new_config, key.split('.'), new_val)
            else:
                new_config[key] = new_val
            config_diff[key] = new_val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _vary_config_control(base_config, config_ranges):
    """Return control configurations.

    Each config_range is gone through sequentially. The base_config is
    trained only once.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 independently
        config_diffs: a list of config diff from base_config
    """
    keys = list(config_ranges.keys())
    # Remove the baseconfig value from the config_ranges
    new_config_ranges = {}
    for key, val in config_ranges.items():
        base_config_val = getattr(base_config, key)
        new_config_ranges[key] = [v for v in val if v != base_config_val]

    # Unravel the input index
    dims = [len(new_config_ranges[k]) for k in keys]
    n_max = int(np.sum(dims))

    configs, config_diffs = list(), list()
    configs.append(deepcopy(base_config))
    config_diffs.append({})

    for i in range(n_max):
        new_config = deepcopy(base_config)

        index = i
        for j, dim in enumerate(dims):
            if index >= dim:
                index -= dim
            else:
                break

        config_diff = dict()
        key = keys[j]

        new_val = config_ranges[key][i]
        if '.' in key:
            nested_set(new_config, key.split('.'), new_val)
        else:
            new_config[key] = new_val
        config_diff[key] = new_val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def islikemodeldir(d):
    """Check if directory looks like a model directory."""
    try:
        files = os.listdir(d)
    except NotADirectoryError:
        return False
    for file in files:
        if ('model.ckpt' in file or 'event' in file or
                'model.pkl' in file or 'model.pt' in file):
            return True
    return False


def _get_alldirs(dir, model, sort):
    """Return sorted model directories immediately below path.

    Args:
        model: bool, if True find directories containing model files
        sort: bool, if True, sort directories by name
    """
    dirs = os.listdir(dir)
    if model:
        dirs = [d for d in dirs if islikemodeldir(os.path.join(dir, d))]
        if islikemodeldir(dir):  # if root is mode directory, return it
            return [dir]
    if sort:
        ixs = np.argsort([int(n) for n in dirs])  # sort by epochs
        dirs = [os.path.join(dir, dirs[n]) for n in ixs]
    return dirs


def select_modeldirs(modeldirs, select_dict=None, acc_min=None):
    """Select model directories.

    Args:
        modeldirs: list of model directories
        select_dict: dict, config must match select_dict to be selected
        acc_min: None or float, minimum validation acc to be included
    """
    new_dirs = []
    for d in modeldirs:
        selected = True
        if select_dict is not None:
            config = load_config(d)  # epoch modeldirs have no configs
            config = flatten_nested_dict(config)
            for key, val in select_dict.items():
                if config[key] != val:
                    selected = False
                    break

        if acc_min is not None:
            log = load_log(d)
            if log['val_acc'][-1] < acc_min:
                selected = False

        if selected:
            new_dirs.append(d)

    return new_dirs


def exclude_modeldirs(modeldirs, exclude_dict=None):
    """Exclude model directories."""
    new_dirs = []
    for d in modeldirs:
        excluded = False
        if exclude_dict is not None:
            config = load_config(d)  # epoch modeldirs have no configs
            config = flatten_nested_dict(config)
            for key, val in exclude_dict.items():
                if config[key] == val:
                    excluded = True
                    break

        if not excluded:
            new_dirs.append(d)

    return new_dirs


def get_modeldirs(path, select_dict=None, exclude_dict=None, acc_min=None):
    dirs = _get_alldirs(path, model=True, sort=True)
    dirs = select_modeldirs(dirs, select_dict=select_dict, acc_min=acc_min)
    dirs = exclude_modeldirs(dirs, exclude_dict=exclude_dict)
    return dirs


def get_experiment_name(model_path):
    """Get experiment name for saving."""
    if islikemodeldir(model_path):
        config = load_config(model_path)
        experiment_name = config.experiment_name
        if experiment_name is None:
            # model_path is assumed to be experiment_name/model_name
            experiment_name = os.path.normpath(model_path).split(os.path.sep)[-2]
    else:
        # Assume this is path to experiment
        experiment_name = os.path.split(model_path)[-1]

    return experiment_name


def get_model_name(model_path):
    """Get model name for saving."""
    if islikemodeldir(model_path):
        config = load_config(model_path)
        model_name = config.get('model_name', None)
        if model_name is None:
            # model_path is assumed to be experiment_name/model_name
            model_name = os.path.split(model_path)[-1]
    else:
        # Assume this is path to experiment
        model_name = os.path.split(model_path)[-1]

    return model_name


def load_pickle(dir, var):
    """Load pickle by epoch in sorted order."""
    out = []
    dirs = get_modeldirs(dir)
    for i, d in enumerate(dirs):
        model_dir = os.path.join(d, 'model.pkl')
        with open(model_dir, 'rb') as f:
            var_dict = pickle.load(f)
            try:
                cur_val = var_dict[var]
                out.append(cur_val)
            except:
                print(var + ' is not in directory:' + d)
    return out


def load_log(logdir):
    """Load log files from tensorboard format."""
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    # Show all tags in the log file
    scalar_names = event_acc.Tags()['scalars']
    log = dict()
    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    step_nums = [0]
    for name in scalar_names:
        w_times, step_nums, vals = zip(*event_acc.Scalars(name))
        log[name] = np.array(vals)
    log['steps'] = np.array(step_nums)

    return log


def nested_set(dic, keys, value):
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def nested_update(dic, new_dic):
    """Nested update of dic with new_dic.

    If dic is {'a': 1, 'b': {'c': 3, 'd': 4}}
    new_dic is {'b': {'d': 5}}

    dic.update(**new_dic) will return
    {'a': 1, 'b': {'d': 5}}

    nested_update(dic, new_dic) will return
    {'a': 1, 'b': {'c': 3, 'd': 5}}
    """
    if isinstance(new_dic, dict):
        dic = deepcopy(dic)
        for key, val in new_dic.items():
            if key in dic:
                dic[key] = nested_update(dic[key], val)
            else:
                dic[key] = nested_update({}, val)
    else:
        dic = new_dic
    return dic


def flatten_nested_dict(dic):
    """Flatten a nested dictionary.

    dic = {'a': {'b': 1}}
    will be flattened as
    new_dic = {'a.b': 1}
    """
    new_dic = dict()
    for key, val in dic.items():
        if isinstance(val, dict):
            tmp_dict = flatten_nested_dict(val)
            for new_key, new_val in tmp_dict.items():
                new_dic[key+'.'+new_key] = new_val
        else:
            new_dic[key] = val
    return new_dic


def load_results(rootpath, get_last=True, idx=None,
                 select=None, exclude=None):
    """Load results from path.

    Args:
        rootpath: root path of all models loading results from
        get_last: boolean, if True return only last step results
        idx: int or None, if int, return results from index idx
        select: dictionary or None, select files that match this dictionary
        exclude: dictionary or None, exclude files that match this dictionary

    Returns:
        res: dictionary of numpy arrays, containing information from all models
    """
    dirs = get_modeldirs(rootpath)
    dirs = select_modeldirs(dirs, select_dict=select)
    dirs = exclude_modeldirs(dirs, exclude_dict=exclude)

    res = defaultdict(list)
    for i, d in enumerate(dirs):
        log = load_log(d)
        config = load_config(d)
        config = flatten_nested_dict(config)
        # print(list(config.keys()))

        # TODO: This is hacky, fix!
        len_log = len(log['loss_train'])  # log need to have loss train then

        # Add logger values
        for key, val in log.items():
            if len(val) == len_log:
                if get_last:
                    val = val[-1]  # store last value in log
                elif idx is not None:
                    val = val[idx]
            res[key].append(val)

            if 'loss' in key:
                res['log_' + key].append(np.log(val))

        for key, val in config.items():
            res[key].append(val)

    skipped_keys = []
    for key, val in res.items():
        try:
            res[key] = np.array(val)
        except:
            skipped_keys.append(key)
    if len(skipped_keys) > 0:
        print('\nSkipped plotting the following parameters:')
        print(skipped_keys)
        print()

    return res


nicename_dict = {
    'loss_train': 'Train loss',
    'acc_train': 'Train accuracy',
    'rnn_eta': 'Eta',
    'lstm': 'LSTM',
    'plastic': 'Plastic',
    'network': 'Network type',
    'steps': 'Training steps',
    'plastic_input': 'Plastic input',
    'plastic_rec': 'Plastic recurrent',
    'use_global_thirdfactor': 'Use Third Factor',
    'hebb_mode': 'Hebbian mode',
    'mlp': 'MLP',
    'pointwise': 'Pointwise',
    'i2h': 'Input-to-hidden',
    'h2o': 'Hidden-to-output',
    'rnn': 'RNN',
    'special_plastic': 'Feedforward plastic',
    'special_plastic_reference': 'Designed feedforward plastic'
}


def nicename(name, mode='dict'):
    """Return nice name for publishing."""
    if mode == 'lr':
        return np.format_float_scientific(name, precision=0, exp_digits=1)
    elif 'acc' in mode:
        return '{:0.2f}'.format(name)
    elif isinstance(name, np.float):
        return '{:0.3f}'.format(name)
    elif isinstance(name, str) and '.' in name:
        parts = name.split('.')
        return ' '.join([nicename(part) for part in parts])
    try:
        return nicename_dict[name]
    except KeyError:
        return str(name)


# colors from https://visme.co/blog/color-combinations/ # 14
blue = np.array([2, 148, 165]) / 255.
red = np.array([193, 64, 61]) / 255.
gray = np.array([167, 156, 147]) / 255.
darkblue = np.array([3, 53, 62]) / 255.
green = np.array([65, 89, 57]) / 255.  # From # 24


def add_colorannot(fig, rect, labels, colors=None,
                   texts=None, orient='horizontal'):
    """Plot color indicating groups"""
    ax = fig.add_axes(rect)
    for il, l in enumerate(np.unique(labels)):
        if colors is None:
            raise NotImplementedError
        else:
            color = colors[il]
        ind_l = np.where(labels == l)[0]
        if (ind_l[-1] - ind_l[0] + 1) == len(ind_l):
            # Only plot if consequtive
            ind_l = [ind_l[0], ind_l[-1] + 1]
            if orient == 'horizontal':
                ax.plot(ind_l, [0, 0], linewidth=4, solid_capstyle='butt',
                        color=color)
                if texts is not None:
                    ax.text(np.mean(ind_l), -1, texts[il], fontsize=7,
                            ha='center', va='top', color=color)
            else:
                ax.plot([0, 0], ind_l, linewidth=4, solid_capstyle='butt',
                        color=color)
                if texts is not None:
                    ax.text(-1, np.mean(ind_l), texts[il], fontsize=7,
                            ha='right', va='center', color=color,
                            rotation='vertical')
        else:
            # If non-consequtive
            if orient == 'horizontal':
                for j in ind_l:
                    ax.plot([j, j+1], [0, 0], linewidth=4,
                            solid_capstyle='butt', color=color)
            else:
                for j in ind_l:
                    ax.plot([0, 0], [j, j+1], linewidth=4,
                            solid_capstyle='butt', color=color)

    if orient == 'horizontal':
        ax.set_xlim([0, len(labels)])
        ax.set_ylim([-1, 1])
    else:
        ax.set_ylim([0, len(labels)])
        ax.set_xlim([-1, 1])
    ax.axis('off')
