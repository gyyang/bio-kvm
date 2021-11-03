"""List of experiments and experiment-specific analysis functions.

Each experiment is a function that returns
    a config dictionary
    a dictionary of hyperparameters to vary
    a str describing the mode to vary these hyperparameters

Each analysis must be named name_analysis, where name is the name of an
experiment.
"""

from collections import OrderedDict
import itertools

import configs
import tools

from copy import deepcopy


def standard_plastic():
    """Standard training setting"""
    fullconfig = configs.get_config('emergeplasticity')
    fullconfig['train_steps'] = 4001

    config = fullconfig['plasticnet']
    config['hebb']['weight'] = False
    config['hebb']['hebb_strength'] = 'scalar'

    fullconfig['dataset']['stim_dim'] = 30

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [0]
    return fullconfig, config_ranges, 'combinatorial'


def standard_plastic_analysis(path):
    import analysis.analysis_longrecall as analysis_longrecall
    # Evaluate network post training
    analysis_longrecall.evaluate_longrun(path, T=500)
    # Plot acc vs memory age
    analysis_longrecall.plot_interval_acc(path)


def hebb_strength():
    """Compare if hebb strength should be scalar or tensor."""
    fullconfig = configs.get_config('emergeplasticity')

    fullconfig['train_steps'] = 4001

    config = fullconfig['plasticnet']
    config['network'] = 'special_plastic'
    config['hidden_size'] = 20

    fullconfig['dataset']['T_min'] = 10
    fullconfig['dataset']['T_max'] = 20
    fullconfig['dataset']['stim_dim'] = 20
    fullconfig['dataset']['p_recall'] = 1.

    config_ranges = OrderedDict()
    config_ranges['plasticnet.i2h.hebb_strength'] = ['scalar', 'tensor']
    config_ranges['plasticnet.h2o.hebb_strength'] = ['scalar', 'tensor']
    return fullconfig, config_ranges, 'sequential'


def random_seed6():
    """Test training setting"""
    fullconfig = configs.get_config('emergeplasticity')
    fullconfig['train_steps'] = 4001
    fullconfig['print_every_steps'] = 200

    config = fullconfig['plasticnet']
    config['network'] = 'special_plastic'
    config['hebb']['mode'] = 'affine'
    config['use_global_thirdfactor'] = True
    config['hidden_size'] = 40

    config_ranges = OrderedDict()
    config_ranges['dummy'] = list(range(5))
    return fullconfig, config_ranges, 'combinatorial'


def p_random():
    """"""
    config = configs.get_config('emergeplasticity')
    config['train_steps'] = 4001
    config['network'] = 'special_plastic'
    config['hebb_mode'] = 'affine'
    config['use_global_thirdfactor'] = True

    config['local_thirdfactor_mode'] = 'random'

    config['T_min'] = 20
    config['T_max'] = 40
    config['stim_dim'] = 30
    config['hidden_size'] = 40

    config_ranges = OrderedDict()
    config_ranges['local_thirdfactor_prob'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    return config, config_ranges, 'combinatorial'


def network_architecture():
    """Standard training setting"""
    config = configs.get_config('emergeplasticity')
    config['train_steps'] = 20001

    config['T_min'] = 20
    config['T_max'] = 40
    config['stim_dim'] = 30
    config['hidden_size'] = 40

    # Parameters for special plastic
    config['plasticnet.hebb_config.mode'] = 'affine'
    config['plasticnet.use_global_thirdfactor'] = True
    config['plasticnet.local_thirdfactor_mode'] = 'random'
    config['plasticnet.local_thirdfactor_prob'] = 0.1

    config_ranges = OrderedDict()
    config_ranges['network'] = ['lstm', 'plastic', 'special_plastic_reference',
                            'special_plastic']
    return config, config_ranges, 'combinatorial'


def affine_init():
    """"""
    config = configs.get_config('emergeplasticity')
    config['train_steps'] = 4001
    # config['train_steps'] = 2001
    config['print_every_steps'] = 200
    config['network'] = 'special_plastic'
    config['hebb_config']['mode'] = 'affine'
    config['use_global_thirdfactor'] = True
    config['local_thirdfactor_mode'] = 'random'
    config['local_thirdfactor_prob'] = 0.1

    config['T_min'] = 20
    config['T_max'] = 40
    config['stim_dim'] = 30
    config['hidden_size'] = 40

    # config['T_min'] = 15
    # config['T_max'] = 25
    # config['stim_dim'] = 8
    # config['hidden_size'] = 20

    config_ranges = OrderedDict()
    config_ranges['hebb_config.init_affine_weight'] = [1.0, 0.5, 0.1]
    config_ranges['hebb_config.init_affine_bias'] = [0, 0.5, 1.0]
    return config, config_ranges, 'combinatorial'


def plastic_parts():
    """Vary input and recurrent plasticity."""
    config = configs.get_config('emergeplasticity')
    config['train_steps'] = 10001
    config['network'] = 'plastic'

    config_ranges = OrderedDict()
    config_ranges['plastic_input'] = [True, False]
    config_ranges['plastic_rec'] = [True, False]
    return config, config_ranges, 'combinatorial'


def thirdfactor():
    """Test third factor"""
    config = configs.get_config('emergeplasticity')
    config['train_steps'] = 10001
    config['network'] = 'plastic'
    # A dataset where not everything needs to be remembered
    config['T'] = 20
    config['p_recall'] = 0.25

    config_ranges = OrderedDict()
    config_ranges['use_global_thirdfactor'] = [False, True]
    return config, config_ranges, 'combinatorial'


def hebb_mode():
    """Test Hebb mode"""
    config = configs.get_config('emergeplasticity')
    config['train_steps'] = 10001
    config['network'] = 'plastic'

    config_ranges = OrderedDict()
    config_ranges['hebb_config.mode'] = [None, 'mlp']
    return config, config_ranges, 'combinatorial'


def thinking_step():
    """Test thinking step"""
    config = configs.get_config('emergeplasticity')
    config['train_steps'] = 10001
    config['network'] = 'plastic'

    config_ranges = OrderedDict()
    config_ranges['thinking_step'] = [1, 2, 3, 4, 5]
    return config, config_ranges, 'combinatorial'


def predictivegate():
    """Test training setting"""
    fullconfig = configs.get_config('emergeplasticity')

    fullconfig['train_steps'] = 4001
    fullconfig['debug'] = False

    config = fullconfig['plasticnet']
    config['network'] = 'special_plastic'
    # config['network'] = 'special_plastic_reference'
    # config['network'] = 'plastic'
    # config['network'] = 'lstm'
    # config['plastic_input'] = False
    config['hebb']['mode'] = 'affine'
    config['use_global_thirdfactor'] = True
    config['hidden_size'] = 40

    dataset_config = fullconfig['dataset']
    dataset_config['T_min'] = 3
    dataset_config['T_max'] = 6
    dataset_config['p_recall'] = 1
    dataset_config['n_repeat'] = 10
    dataset_config['sigma'] = 0.1

    config_ranges = OrderedDict()
    config_ranges['plasticnet.predictive_gate'] = [False, True]
    return fullconfig, config_ranges, 'combinatorial'


def scalable():
    """Scalable network. No parameters are tied to network size."""
    fullconfig = configs.get_config('emergeplasticity')

    fullconfig['train_steps'] = 4001

    config = fullconfig['plasticnet']
    config['hebb']['weight'] = False
    config['hebb']['hebb_strength'] = 'scalar'

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [0]
    return fullconfig, config_ranges, 'combinatorial'


def scalable_analysis(save_path):
    from analysis.analysis_hebb import evaluate_network_size, plot_network_size_acc
    save_path = tools.get_modeldirs(save_path)[0]
    evaluate_network_size(save_path)
    plot_network_size_acc(save_path)


def compare_hopfield():
    """Train networks to compare with Hopfield.

    Assuming input dimension the same as hidden dimension
    """
    fullconfig = configs.SLTrainConfig
    fullconfig['dataset'] = configs.RecallDatasetConfig
    fullconfig['plasticnet'] = configs.SpecialPlasticRNNConfig

    fullconfig['train_steps'] = 4001
    fullconfig['dataset']['p_recall'] = 1.0

    config = fullconfig['plasticnet']
    config['i2h']['weight'] = False
    config['h2o']['weight'] = False
    config['i2h']['hebb_strength'] = 'scalar'
    config['h2o']['hebb_strength'] = 'scalar'

    hidden_sizes = [20, 40, 60]
    config_ranges = OrderedDict()
    config_ranges['plasticnet.hidden_size'] = hidden_sizes
    config_ranges['dataset.stim_dim'] = hidden_sizes
    config_ranges['dataset.T_min'] = [int(h/2) for h in hidden_sizes]
    config_ranges['dataset.T_max'] = hidden_sizes

    # At each time step, on average 1.5 neurons active
    # config_ranges['plasticnet.local_thirdfactor_prob'] = [1.5/h for h in
    #                                                       hidden_sizes]
    return fullconfig, config_ranges, 'sequential'


def compare_hopfield_analysis(path):
    del path
    import analysis.analysis_acc as analysis_acc
    # results = analysis_acc.compute_acc_vs_stim_dim_hopfield()
    # results = analysis_acc.compute_acc_vs_stim_dim_special_plastic_reference()
    # results = analysis_acc.compute_acc_vs_stim_dim_special_plastic()
    analysis_acc.plot_acc_vs_stim_dim(['hopfield', 'special_plastic', 'special_plastic_reference'])


def heteroassociative():
    fullconfig = deepcopy(configs.SLTrainConfig)
    fullconfig['training'] = False
    fullconfig['print_every_steps'] = 30

    fullconfig['dataset'] = deepcopy(configs.RecallDatasetConfig)
    fullconfig['dataset']['stim_dim'] = 40
    fullconfig['dataset']['p_recall'] = 1
    fullconfig['dataset']['heteroassociative'] = True
    fullconfig['dataset']['heteroassociative_stim_dim'] = 20
    fullconfig['input_size'] = 40 + 1
    fullconfig['output_size'] = 20

    fullconfig['plasticnet'] = deepcopy(configs.SpecialPlasticRNNReferenceConfig)
    fullconfig['plasticnet']['residual_conn'] = False
    fullconfig['plasticnet']['i2h']['zero_in_detach'] = True
    fullconfig['plasticnet']['h2o']['zero_in_detach'] = True
    fullconfig['plasticnet']['hidden_size'] = 40
    fullconfig['plasticnet']['recalc_output_in_forward'] = True

    fullconfig['train_steps'] = 31

    config_ranges = OrderedDict()
    config_ranges['dummy'] = ['0']

    return fullconfig, config_ranges, 'sequential'


def heteroassociative_random():
    fullconfig, config_ranges, config_type = heteroassociative()
    fullconfig['train_steps'] = 20000

    fullconfig['plasticnet'] = deepcopy(configs.SpecialPlasticRNNConfig)
    fullconfig['plasticnet']['recalc_output_in_forward'] = True
    fullconfig['plasticnet']['i2h']['zero_in_detach'] = True
    fullconfig['plasticnet']['h2o']['zero_in_detach'] = True
    fullconfig['plasticnet']['local_thirdfactor']['mode'] = 'random'
    fullconfig['plasticnet']['local_thirdfactor']['b0'] = 4/40.

    return fullconfig, config_ranges, config_type


def heteroassociative_random_reference():
    fullconfig, config_ranges, config_type = heteroassociative()

    random_net_config = deepcopy(configs.get_config('ref_rand'))
    fullconfig.pop('plasticnet', None)
    fullconfig['plasticnet'] = random_net_config['plasticnet']
    fullconfig['plasticnet']['recalc_output_in_forward'] = True
    fullconfig['plasticnet']['i2h']['zero_in_detach'] = True
    fullconfig['plasticnet']['h2o']['zero_in_detach'] = True

    return fullconfig, config_ranges, config_type


def heteroassociative_hopfield():
    fullconfig, config_ranges, config_type = heteroassociative()

    fullconfig.pop('plasticnet', None)
    fullconfig['plasticnet'] = {}
    fullconfig['plasticnet']['hopfield_config'] = deepcopy(configs.HopfieldConfig)
    fullconfig['plasticnet']['hopfield_config']['learn_params'] = True
    fullconfig['plasticnet']['hopfield_config']['steps'] = 1
    fullconfig['plasticnet']['network'] = 'hopfield'
    fullconfig['plasticnet']['hidden_size'] = -1
    fullconfig['plasticnet']['thinking_step'] = 1
    fullconfig['plasticnet']['residual_conn'] = False

    return fullconfig, config_ranges, config_type


def seqrecall():
    fullconfig = configs.get_config('seqrecall')
    fullconfig['training'] = False
    fullconfig['train_steps'] = 300001
    fullconfig['lr'] = 1E-4

    fullconfig['dataset']['pattern_dim'] = 40

    fullconfig['ctrlnet']['ctrl_type'] = None
    fullconfig['ctrlnet']['ctrl_size'] = fullconfig['dataset']['pattern_dim']
    fullconfig['ctrlnet']['learn_modu'] = False
    fullconfig['ctrlnet']['activ'] = 'identity'

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [0]
    return fullconfig, config_ranges, 'combinatorial'


def seqrecall_random():
    fullconfig, config_ranges, config_type = seqrecall()
    fullconfig['training'] = True
    fullconfig.pop('plasticnet')
    fullconfig['plasticnet'] = deepcopy(configs.SpecialPlasticRNNConfig)
    fullconfig['plasticnet']['recalc_output_in_forward'] = True
    fullconfig['plasticnet']['i2h']['zero_in_detach'] = True
    fullconfig['plasticnet']['h2o']['zero_in_detach'] = True
    fullconfig['plasticnet']['local_thirdfactor']['mode'] = 'random'
    fullconfig['plasticnet']['local_thirdfactor']['b0'] = 4/40.

    return fullconfig, config_ranges, config_type


def seqrecall_random_reference():
    fullconfig, config_ranges, config_type = seqrecall()
    fullconfig['training'] = False
    fullconfig.pop('plasticnet')
    fullconfig['plasticnet'] = deepcopy(configs.SpecialPlasticRNNReferenceConfig)
    fullconfig['plasticnet']['recalc_output_in_forward'] = True
    fullconfig['plasticnet']['i2h']['zero_in_detach'] = True
    fullconfig['plasticnet']['h2o']['zero_in_detach'] = True
    fullconfig['plasticnet']['local_thirdfactor']['mode'] = 'random'
    fullconfig['plasticnet']['local_thirdfactor']['b0'] = 4/40.

    return fullconfig, config_ranges, config_type


def seqrecall_hopfield():
    fullconfig, config_ranges, config_type = seqrecall()
    fullconfig['training'] = True
    fullconfig.pop('plasticnet', None)
    fullconfig['plasticnet'] = {}
    fullconfig['plasticnet']['hopfield_config'] = deepcopy(configs.HopfieldConfig)
    fullconfig['plasticnet']['hopfield_config']['learn_params'] = True
    fullconfig['plasticnet']['hopfield_config']['steps'] = 1
    fullconfig['plasticnet']['network'] = 'hopfield'
    fullconfig['plasticnet']['hidden_size'] = -1

    return fullconfig, config_ranges, config_type


def copy():
    fullconfig = deepcopy(configs.get_config('copypaste_ctrl'))
    fullconfig['train_steps'] = 600001
    fullconfig['lr'] = 1E-3

    fullconfig['ctrlnet']['ctrl_type'] = 'FFSigmoid'
    fullconfig['ctrlnet']['ctrl_size'] = 40
    fullconfig['ctrlnet']['learn_modu'] = True
    fullconfig['ctrlnet']['complicated_modu'] = True

    fullconfig['dataset']['modu_input'] = False
    fullconfig['dataset']['balanced'] = True
    fullconfig['dataset']['n_patterns_max'] = 10
    fullconfig['dataset']['pattern_dim'] = 25
    fullconfig['dataset']['n_paste_max'] = 1

    mem_config = fullconfig['plasticnet']
    mem_config['hidden_size'] = 40

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [0]

    fullconfig['ctrlnet']['train_memnet'] = False

    return fullconfig, config_ranges, 'combinatorial'

def copy_random():
    fullconfig, config_ranges, config_type = copy()

    fullconfig.pop('plasticnet', None)
    fullconfig['plasticnet'] = deepcopy(configs.SpecialPlasticRNNConfig)
    fullconfig['plasticnet']['recalc_output_in_forward'] = True
    fullconfig['plasticnet']['i2h']['zero_in_detach'] = True
    fullconfig['plasticnet']['h2o']['zero_in_detach'] = True

    fullconfig['ctrlnet']['train_memnet'] = True

    return fullconfig, config_ranges, config_type


def copy_random_reference():
    fullconfig, config_ranges, config_type = copy()

    random_net_config = configs.get_config('ref_rand')
    fullconfig.pop('plasticnet', None)
    fullconfig['plasticnet'] = random_net_config['plasticnet']
    fullconfig['plasticnet']['recalc_output_in_forward'] = True
    fullconfig['plasticnet']['i2h']['zero_in_detach'] = True
    fullconfig['plasticnet']['h2o']['zero_in_detach'] = True

    fullconfig['ctrlnet']['train_memnet'] = False

    return fullconfig, config_ranges, config_type


def copy_hopfield():
    fullconfig, config_ranges, config_type = copy()
    fullconfig.pop('plasticnet', None)
    fullconfig['plasticnet'] = {}
    fullconfig['plasticnet']['hopfield_config'] = deepcopy(configs.HopfieldConfig)
    fullconfig['plasticnet']['hopfield_config']['learn_params'] = False
    fullconfig['plasticnet']['hopfield_config']['steps'] = 1
    fullconfig['plasticnet']['hopfield_config']['take_sign_in_output'] = False
    fullconfig['plasticnet']['hopfield_config']['learning_rate'] = 1/40.

    fullconfig['plasticnet']['recalc_output_in_forward'] = True
    fullconfig['plasticnet']['network'] = 'hopfield'
    fullconfig['plasticnet']['hidden_size'] = -1
    fullconfig['plasticnet']['thinking_step'] = 1
    fullconfig['plasticnet']['residual_conn'] = False

    return fullconfig, config_ranges, config_type


def copypaste():
    fullconfig = configs.get_config('copypaste_ctrl')
    fullconfig['train_steps'] = 600001
    fullconfig['lr'] = 1E-4
    fullconfig['debug'] = False

    fullconfig['ctrlnet']['ctrl_type'] = 'FF'
    fullconfig['ctrlnet']['ctrl_size'] = 160
    fullconfig['ctrlnet']['learn_modu'] = True
    fullconfig['ctrlnet']['complicated_modu'] = True
    fullconfig['ctrlnet']['activ'] = 'identity'

    fullconfig['dataset']['modu_input'] = False
    fullconfig['dataset']['balanced'] = True
    fullconfig['dataset']['n_patterns_max'] = 5
    fullconfig['dataset']['pattern_dim'] = 8
    fullconfig['dataset']['n_paste_max'] = 5

    mem_config = fullconfig['plasticnet']
    mem_config['hidden_size'] = 40

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [0]

    return fullconfig, config_ranges, 'combinatorial'


def recall_spatial_corr():
    fullconfig = configs.get_config('emergeplasticity')

    config_ranges = OrderedDict()
    config_ranges['dataset.spatial_corr'] = [0, 0.3, 0.6, 0.9]
    config_ranges['plasticnet'] = [configs.PlasticNetworkConfig, configs.LSTMConfig]
    return fullconfig, config_ranges, 'combinatorial'


def recall_spatial_corr_analysis(path):
    from analysis.analysis import plot_progress
    plot_progress(path, ykeys='all', legend_key='dataset.spatial_corr',
                  exclude={'plasticnet.network':'lstm'})


def trainable_weights_hebbness():
    fullconfig = configs.get_config('emergeplasticity')

    config_ranges = OrderedDict()
    config_ranges['plasticnet.trainable_weights'] = [False, True]

    return fullconfig, config_ranges, 'sequential'


def passive_decay():
    fullconfig = configs.get_config('emergeplasticity')
    fullconfig['plasticnet'] = configs.SpecialPlasticRNNConfig
    fullconfig['train_steps'] = 40000
    fullconfig['converged_loss_decrease_frac'] = 0.001
    fullconfig['converged_acc_thres'] = 0.99

    hidden_size = fullconfig['plasticnet']['hidden_size']
    T_max_list = [int(k*hidden_size) for k in (0.75, 1.5, 3)]
    T_min_list = [int(0.9*T) for T in T_max_list]
    decays_list = [None, 'active', 'passive_global', 'passive_neuron', 'passive_synapse']

    config_ranges = OrderedDict()
    config_ranges['dataset.T_max'] = T_max_list * len(decays_list)
    config_ranges['dataset.T_min'] = T_min_list * len(decays_list)
    config_ranges['plasticnet.h2o.decay'] = [d for d in decays_list
                                               for _ in range(len(T_max_list))]

    return fullconfig, config_ranges, 'sequential'


def passive_decay_tensor_strength():
    fullconfig, config_ranges, mode = passive_decay()
    fullconfig['plasticnet']['h2o']['hebb_strength'] = 'tensor'
    fullconfig['plasticnet']['i2h']['hebb_strength'] = 'tensor'
    return fullconfig, config_ranges, mode


def passive_decay_analysis(path):
    from analysis.analysis import plot_progress
    plot_progress(path, ykeys='all', legend_key='plasticnet.h2o.decay',
                  exclude={'plasticnet.reset_to_reference':True})


def tvt_test_fixed_T():
    fullconfig = configs.get_config('emergeplasticity')

    fullconfig['train_steps'] = 4000
    fullconfig['dataset']['T_min'] = 40
    fullconfig['dataset']['T_max'] = 40
    fullconfig['dataset']['n_repeat'] = 1
    fullconfig['dataset']['chance'] = 0.7
    fullconfig['dataset']['p_recall'] = 1.0
    fullconfig['dataset']['stim_dim'] = 40
    fullconfig['dataset']['balanced'] = True
    config = fullconfig['plasticnet']
    config['network'] = 'tvt'
    config_ranges = OrderedDict()
    config_ranges['dummy'] = [0]
    return fullconfig, config_ranges, 'combinatorial'

def tvt():
    fullconfig = configs.get_config('emergeplasticity')

    fullconfig['train_steps'] = 16001

    config = fullconfig['plasticnet']
    config['network'] = 'tvt'

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [0]
    return fullconfig, config_ranges, 'combinatorial'

def train_parameterized_capacity():
    """
    Train parameterized PlasticNet, varying hidden dimension, with SGD and CMA-ES
    to empirically estimate capacity
    """
    fullconfig = configs.get_config('emergeplasticity')
    fullconfig['seed'] = None

    fullconfig['train_steps'] = 10000
    fullconfig['print_every_steps'] = 50

    params = []
    for size in [20, 40, 60, 100]:
        for train in ['evolve', 'supervisedtrain']:
            for ltf in ['sequential', 'random_linear_dynamics', 'random']:
                if train == 'supervisedtrain' and ltf.startswith('random'):
                    continue
                params.append((train, size, ltf))
    train, size, ltf = zip(*params)

    config_ranges = OrderedDict()
    config_ranges['plasticnet.local_thirdfactor.mode'] = ltf
    config_ranges['train_type'] = train
    config_ranges['plasticnet.hidden_size'] \
    = config_ranges['dataset.stim_dim'] \
    = config_ranges['dataset.T_min'] \
    = config_ranges['dataset.T_max'] \
    = size

    return fullconfig, config_ranges, 'sequential'


def train_random_capacity():
    """
    Train parameterized PlasticNet, varying hidden dimension, with SGD and CMA-ES
    to empirically estimate capacity
    """
    fullconfig = configs.get_config('emergeplasticity')

    fullconfig['train_steps'] = 10000
    fullconfig['print_every_steps'] = 50

    params = []
    for size in [20, 40, 80, 160]:
        for prob in [k/size for k in (1,2,4,8,16)]:
            params.append((size,prob))
    size,prob = zip(*params)

    config_ranges = OrderedDict()
    config_ranges['plasticnet.local_thirdfactor.b0'] \
        = config_ranges['plasticnet.local_thirdfactor.b1'] = prob
    config_ranges['plasticnet.hidden_size'] = config_ranges['dataset.stim_dim'] = size
    config_ranges['dataset.T_min'] = [s/2 for s in size]
    config_ranges['dataset.T_max'] = [2*s for s in size]

    return fullconfig, config_ranges, 'sequential'


def train_corr():
    N = 40
    config = configs.get_config('emergeplasticity', stim_dim=N, hidden_size=N)
    config['print_every_steps'] = 50
    config['train_steps'] = 15000
    config['save_path'] = './files/train_corr/'
    config['plasticnet']['i2h']['init_affine_weight'] = 1e-2
    config['plasticnet']['i2h']['init_affine_bias'] = 1e-2
    config['plasticnet']['h2o']['init_affine_weight'] = 1e-2
    config['plasticnet']['h2o']['init_affine_bias'] = 1e-2
    config['dataset']['T_min'] = N/2
    config['dataset']['T_max'] = 2*N
    config['dataset']['temporal_corr_mode']= 'template'

    config_ranges = OrderedDict()
    config_ranges['dataset.temporal_corr'] = [0.3, 0.6, 0.9]
    return config, config_ranges, 'sequential'


####################
### "Unit" tests ###
####################

def test_load_results_variable_length():
    fullconfig = configs.get_config('emergeplasticity')
    fullconfig['print_every_steps'] = 1

    config_ranges = OrderedDict()
    config_ranges['train_steps'] = [5, 6]
    return fullconfig, config_ranges, 'combinatorial'


def test_plasticnet_config():
    fullconfig = configs.get_config('emergeplasticity')
    fullconfig['print_every_steps'] = 1
    fullconfig['train_steps'] = 5

    config_ranges = OrderedDict()
    config_ranges['plasticnet'] = [
        configs.SpecialPlasticRNNConfig,
        configs.SpecialPlasticRNNReferenceConfig,
        configs.LSTMConfig]

    return fullconfig, config_ranges, 'combinatorial'


def test_passive_decay():
    fullconfig = configs.get_config('emergeplasticity')
    fullconfig['print_every_steps'] = 1
    fullconfig['train_steps'] = 5
    fullconfig['plasticnet'] = configs.SpecialPlasticRNNConfig

    config_ranges = OrderedDict()
    config_ranges['plasticnet.h2o.decay'] = [True, 'active', 'passive_global',
                                             'passive_neuron', 'passive_synapse',
                                             None, False]

    return fullconfig, config_ranges, 'sequential'


def test_passive_decay_analysis(path):
    from tools import load_config, get_modeldirs
    from models.model_utils import get_model
    from datasets.dataset_utils import get_dataset
    import os
    dirs = get_modeldirs(path)
    for i, d in enumerate(dirs):
        config = load_config(d)

        dataset = get_dataset(config['dataset'], verbose=False)
        input_size = dataset.input_dim
        output_size = dataset.output_dim

        net = get_model(config)
        model_path = os.path.join(config['save_path'], 'model.pt')
        net.load(model_path)
        print(net)
        print('net.rnn.h2o.decay=', net.rnn.h2o.decay)
        print('net.rnn.h2o.decay_rate=', net.rnn.h2o.decay_rate)
        print()


def test_early_stop():
    fullconfig = configs.get_config('emergeplasticity')
    fullconfig['plasticnet'] = configs.SpecialPlasticRNNConfig
    fullconfig['print_every_steps'] = 1
    fullconfig['train_steps'] = 5

    config_ranges = OrderedDict()
    config_ranges['converged_loss_decrease_frac'] = [-float('inf'), 0.5]
    config_ranges['converged_acc_thres'] = [float('inf'), 0.7]

    return fullconfig, config_ranges, 'combinatorial'
