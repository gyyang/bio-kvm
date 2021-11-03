import os
import subprocess
from pathlib import Path

import experiments
import tools
import settings

use_torch = settings.use_torch
FILEPATH = settings.FILEPATH


def master_train(config):
    import training.train as supervisedtrain
    try:
        import training.train_place_cell as placecelltrain
        import training.train_tem as temtrain
    except ImportError:
        print('Error in importing specialized training')
    import training.evolution as evolution

    if config['train_type'] == 'supervisedtrain':
        supervisedtrain.train(config)
    elif config['train_type'] == 'placecelltrain':
        placecelltrain.train(config)
    elif config['train_type'] == 'evolve':
        evolution.evolve(config)
    elif config['train_type'] == 'temtrain':
        temtrain.train(config)
    else:
        raise ValueError('Unknown config train type',
                         config['train_type'])


def local_train(config, path=None, train_arg=None, **kwargs):
    """Train all models locally."""
    if path is None:
        path = './'

    experiment_name = config['experiment_name']
    model_name = config['model_name']
    config['save_path'] = os.path.join(path, 'files', experiment_name,
                                       model_name)
    master_train(config, **kwargs)


def write_jobfile(cmd, jobname, sbatchpath='./sbatch/',
                  nodes=1, ppn=1, gpus=0, mem=16, nhours=3):
    """
    Create a job file.

    Args:
        cmd : str, Command to execute.
        jobname : str, Name of the job.
        sbatchpath : str, Directory to store SBATCH file in.
        scratchpath : str, Directory to store output files in.
        nodes : int, optional, Number of compute nodes.
        ppn : int, optional, Number of cores per node.
        gpus : int, optional, Number of GPU cores.
        mem : int, optional, Amount, in GB, of memory.
        ndays : int, optional, Running time, in days.
        queue : str, optional, Queue name.

    Returns:
        jobfile : str, Path to the job file.
    """

    os.makedirs(sbatchpath, exist_ok=True)
    jobfile = os.path.join(sbatchpath, jobname + '.s')
    # logname = os.path.join('log', jobname)

    with open(jobfile, 'w') as f:
        f.write(
            '#! /bin/sh\n'
            + '\n'
            # + '#SBATCH --nodes={}\n'.format(nodes)
            # + '#SBATCH --ntasks-per-node=1\n'
            # + '#SBATCH --cpus-per-task={}\n'.format(ppn)
            + '#SBATCH --mem-per-cpu={}gb\n'.format(mem)
            # + '#SBATCH --partition=xwang_gpu\n'
            + '#SBATCH --gres=gpu:1\n'
            + '#SBATCH --time={}:00:00\n'.format(nhours)
            # + '#SBATCH --mem=128gb\n'
            # + '#SBATCH --job-name={}\n'.format(jobname[0:16])
            # + '#SBATCH --output={}log/{}.o\n'.format(scratchpath, jobname[0:16])
            + '\n'
            # + 'cd {}\n'.format(scratchpath)
            # + 'pwd > {}.log\n'.format(logname)
            # + 'date >> {}.log\n'.format(logname)
            # + 'which python >> {}.log\n'.format(logname)
            # + '{} >> {}.log 2>&1\n'.format(cmd, logname)
            + cmd + '\n'
            + '\n'
            + 'exit 0;\n'
            )
        print(jobfile)
    return jobfile


def cluster_train(config, path, train_arg=None):
    """Train models on cluster."""
    experiment_name = config['experiment_name']
    model_name = config['model_name']
    config['save_path'] = os.path.join(path, 'files', experiment_name,
                                       model_name)
    os.makedirs(config['save_path'], exist_ok=True)

    # TEMPORARY HACK
    # Hack: assuming data_dir of form './files/XX'
    # config['data_dir'] = os.path.join(path, config['data_dir'][2:])

    tools.save_config(config, config['save_path'])

    arg = '\'' + config['save_path'] + '\''

    cmd = r'''python -c "import training.train as supervisedtrain;supervisedtrain.train_from_path(''' + arg + ''')"'''
    jobfile = write_jobfile(cmd, jobname=experiment_name + '_' + model_name,
                            mem=12)
    subprocess.call(['sbatch', jobfile])


def train_experiment(experiment, use_cluster=False, path=None, train_arg=None,
                     **kwargs):
    """Train model across platforms given experiment name.

    Args:
        experiment: str, name of experiment to be run
            must correspond to a function in experiments.py
        use_cluster: bool, whether to run experiments on cluster
        path: str, path to save models and config
        train_arg: None or str
    """
    if path is None:
        # Default path
        if use_cluster:
            path = settings.cluster_path
        else:
            path = Path('./')

    print('Training {:s} experiment'.format(experiment))
    experiment_files = [experiments]

    experiment_found = False
    for experiment_file in experiment_files:
        if experiment in dir(experiment_file):
            # Get list of configurations from experiment function
            fullconfig, config_ranges, mode = getattr(experiment_file,
                                                      experiment)()
            configs = tools.vary_config(fullconfig, config_ranges, mode)
            experiment_found = True
            break
        else:
            experiment_found = False

    if not experiment_found:
        raise ValueError('Experiment not found: ', experiment)

    for config in configs:
        config['experiment_name'] = experiment
        if use_cluster:
            cluster_train(config, path=path, train_arg=train_arg)
        else:
            local_train(config, path=path, train_arg=train_arg, **kwargs)


def _specific_analyze_experiment(experiment, experiment_files=None):
    """Analyze an experiment with specialized analysis.

    Run specialized functions named experiment_analysis

    Args:
        experiment: str, name of experiment
        experiment_files: optional list of experiment files to search within
    """
    path = os.path.join(FILEPATH, experiment)

    _experiment_files = [experiments]
    if experiment_files is not None:
        _experiment_files = _experiment_files + experiment_files

    # Search for specialized analysis
    experiment_found = False
    for experiment_file in _experiment_files:
        if (experiment + '_analysis') in dir(experiment_file):
            getattr(experiment_file, experiment + '_analysis')(path)
            experiment_found = True
            break
        else:
            experiment_found = False

    if not experiment_found:
        print('Specialized analysis not found for experiment', experiment)


def _generic_analyze_plasticnet(experiment):
    """Attempt generic analysis of plastic net.

    Args:
        experiment: str, name of experiment
    """
    import analysis.analysis as analysis
    import analysis.analysis_hebb as analysis_hebb
    from datasets.dataset_utils import visualize_dataset

    # Use experiment from experiments.name_analysis
    print('Analyzing {:s} experiment'.format(experiment))
    path = os.path.join(FILEPATH, experiment)

    # Get experiment
    run_generic_plasticnet_analysis = False
    try:
        fullconfig, config_ranges, _ = getattr(experiments, experiment)()
        train_type = fullconfig['train_type']
        if train_type == 'supervisedtrain' or train_type == 'temtrain' or train_type == 'evolve':
            run_generic_plasticnet_analysis = True
    except AttributeError:
        print('No training configs for experiment: ' + experiment)

    if run_generic_plasticnet_analysis:
        try:
            visualize_dataset(path)  # Visualize dataset
        except:
            print("Dataset visualization function failed.")

        keys = list(config_ranges.keys())  # the hyperparameters varied
        analysis.plot_progress(path, ykeys='all', legend_key=keys)
        if len(keys) == 1:
            analysis.plot_results(path, xkey=keys[0], ykey='acc_train')

        try:
            analysis_hebb.plot_prepost_hebbprogress(path)
        except:
            print('analysis_hebb.plot_prepost_hebbprogress can not be run')


def analyze_experiment(experiment, general=True, specific=True):
    """Analyze an experiment.

    Attempt to infer the kind of analysis to be done automatically.
    Analyses are separated into standard analysis for one type of network,
    and specialized analyses as defined in experiments.py

    Args:
        experiment: str, name of experiment
        general: bool, if True, run general analysis
        specific: bool, if True, run specific analysis
    """
    if general:
        _generic_analyze_plasticnet(experiment)
    if specific:
        _specific_analyze_experiment(experiment)
