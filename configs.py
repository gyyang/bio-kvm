import os, math
from copy import deepcopy
import tools

N_MEMORY_INPUT = 35
N_MEMORY_HIDDEN = 40

###################
# Dataset configs #
###################
RecallDatasetConfig = {
    'name': 'recall',
    'T_min': 15,
    'T_max': 25,
    'T_distribution': 'uniform',
    'stim_dim': N_MEMORY_INPUT,
    'p_recall': 1,
    'chance': 0.7,
    'balanced': True,
    'spatial_corr': 0.,
    'temporal_corr': 0.,
    'temporal_corr_mode': None, #'template' or 'drift'
    'recall_order': 'sequential', #'interleave'
    'recall_interleave_delay': None, #only for recall_order=='interleave'
    'n_repeat': 1, # repeat each pattern n_repeat times
    'sigma': 0., # noise
}

CopyPasteDatasetConfig = {
    'name': 'copypaste',
    'pattern_dim': 8,
    'n_patterns_min': 1,
    'n_patterns_max': 5,
    'n_paste_min': 1,
    'n_paste_max': 10,
    'modu_input': False
}

SeqRecallDatasetConfig = {
    'name': 'seqrecall'
}


###################
# Network configs #
###################
PlasticLinearConfig = {
        # Whether to clamp hebb weights
        'clamp': float('inf'),
        # Whether previous weights are decayed
        'decay': 'active', #active, passive_global, passive_neuron, passive_synapse, or None
        # Whether the linear layer contains a weight term
        'weight': False,
        # Whether the linear layer contains a bias term
        'bias': False,
        # Whether to include a stability term
        'stability': False,
        # Overall strength of Hebbian weights
        'hebb_strength': None, #'scalar', 'tensor', or None
        # Normalize plastic matrix at each timestep
        'normalize' : False, #row, column, or False
        # If normalize, scale each row/column to this norm instead of 1
        'normalize_scale' : None, #float(N_MEMORY_INPUT),
        # How to compute update
        'mode': 'affine', # 'affine', 'pointwise', 'mlp', None
        # Initial values for affine mode
        'init_affine_weight': 1.0,
        'init_affine_bias': 0.0,
}


SpecialPlasticRNNConfig = {
        'hidden_size': N_MEMORY_HIDDEN,
        # TODO: some code rely on this 'network' field, but that's opaque
        'network': 'special_plastic',
        'thinking_step': 1,  # model_utils.MemoryNet
        'use_global_thirdfactor': True,
        'reset_to_reference': False,
        'local_thirdfactor': {'mode': 'random', #sequential, least_recent, least_recent_random
                              'a1': 0.,
                              'b1': 4./N_MEMORY_HIDDEN,
                              'a0': 0.,
                              'b0': 4./N_MEMORY_HIDDEN,
                              'use_modu_input': False},
        'h2o_use_local_thirdfactor':False,
        'predictive_gate': False, #multiply global_tf by avg prediction error
        'normalize_input': False, #normalize the input vector to norm=1
        'residual_conn' : False, #whether to add the residual connection out += input
        'i2h': PlasticLinearConfig,
        'h2o': tools.nested_update(PlasticLinearConfig, {'decay':'passive_global'})
}


SpecialPlasticRNNReferenceConfig = tools.nested_update(SpecialPlasticRNNConfig,
    {'local_thirdfactor': {'mode':'sequential',
                           'use_modu_input': False},
     'h2o': tools.nested_update(PlasticLinearConfig, {'decay':'active'}),
     'h2o_use_local_thirdfactor' : True, #required for active decay
     'reset_to_reference': True}
)


MemoryConfig = {
    'n_concept': 100,
    'write_mode': 'usage',
    'overwrite': True,
    'softmax': 'divisive',
    'val_plasticity': 'hebb',
}


CtrlrMemConfig = {
    'ctrl_size': 64,
    'ctrl_type': 'LSTM',
    'learn_modu': True,
    'use_global_thirdfactor': True,
    'activ': 'sigmoid'
}


HopfieldConfig = {
    'steps': float('inf'), #inf means always run to fixed point
    'decay_rate' : 1.,
    'learning_rate': 1., #1./math.sqrt(N_MEMORY_INPUT) for continual
    'clamp_val' : float('inf'), #0.35 for continual (Parisi 1986)
    'learn_params' : False,
    'zero_in_detach' : True,
    'take_sign_in_output' : True
}


TVTConfig = {
    'hidden_size': N_MEMORY_HIDDEN,
    'network': 'tvt',
    'thinking_step': 0, #dummy value
}


####################
# Training configs #
####################
SLTrainConfig = {
    'train_type': 'supervisedtrain',
    'save_path': None,
    'train_steps': 4001,
    'print_every_steps': 400,
    'debug': False,
    'lr': 0.001,
    'training': True,
    'converged_loss_decrease_frac' : -float('inf'),
    'converged_acc_thres': float('inf'),
}


EvolutionaryTrainConfig = tools.nested_update(SLTrainConfig,
    {'train_type': 'evolve',
    'print_every_steps': 400,
    'seed': None} #this seed will be used for numpy, pytorch, and pycma
)


def get_config(name, stim_dim=None, hidden_size=None):
    """Get several kinds of pre-packaged configurations."""
    #Note: need to copy everything (maybe deepcopy is overkill?) so that
    #modifying the config doesn't change the original dict
    if name == 'emergeplasticity':
        FullConfig = deepcopy(SLTrainConfig)
        FullConfig['dataset'] = deepcopy(RecallDatasetConfig)
        FullConfig['plasticnet'] = deepcopy(SpecialPlasticRNNConfig)
    elif name == "copypaste_ctrl":
        FullConfig = deepcopy(SLTrainConfig)
        FullConfig['dataset'] = deepcopy(CopyPasteDatasetConfig)
        FullConfig['ctrlnet'] = deepcopy(CtrlrMemConfig)
        FullConfig['plasticnet'] = deepcopy(SpecialPlasticRNNReferenceConfig)
        FullConfig['plasticnet']['recalc_output_in_forward'] = True
        FullConfig['plasticnet']['i2h']['zero_in_detach'] = True
        FullConfig['plasticnet']['h2o']['zero_in_detach'] = True
    elif name == "seqrecall":
        FullConfig = deepcopy(SLTrainConfig)
        FullConfig['dataset'] = deepcopy(SeqRecallDatasetConfig)
        FullConfig['ctrlnet'] = deepcopy(CtrlrMemConfig)
        FullConfig['plasticnet'] = deepcopy(SpecialPlasticRNNReferenceConfig)
        FullConfig['plasticnet']['recalc_output_in_forward'] = True
        FullConfig['plasticnet']['i2h']['zero_in_detach'] = True
        FullConfig['plasticnet']['h2o']['zero_in_detach'] = True
    elif name == 'hopfield':
        FullConfig = {}
        FullConfig['dataset'] = deepcopy(RecallDatasetConfig)
        FullConfig['hopfield'] = deepcopy(HopfieldConfig)
    elif name == "cmaes": #same as 'emergeplasticity' except for EvolutionaryTrainConfig
        FullConfig = deepcopy(EvolutionaryTrainConfig)
        FullConfig['dataset'] = deepcopy(RecallDatasetConfig)
        FullConfig['plasticnet'] = deepcopy(SpecialPlasticRNNConfig)
    elif name == 'tvt_net':
        FullConfig = deepcopy(SLTrainConfig)
        FullConfig['dataset'] = deepcopy(RecallDatasetConfig)
        FullConfig['tvt_memory'] = deepcopy(TVTConfig)
    elif name == 'ref_seq':
        FullConfig = {}
        FullConfig['dataset'] = deepcopy(RecallDatasetConfig)
        FullConfig['plasticnet'] = deepcopy(SpecialPlasticRNNReferenceConfig)
    elif name == 'ref_rand':
        FullConfig = {}
        FullConfig['dataset'] = deepcopy(RecallDatasetConfig)
        FullConfig['plasticnet'] = deepcopy(SpecialPlasticRNNReferenceConfig)
        if hidden_size is None:
            hidden_size = FullConfig['plasticnet']['hidden_size']
        FullConfig['plasticnet']['local_thirdfactor'] = {'mode': 'random',
                                                         'b0': 4/hidden_size,
                                                         'use_modu_input': False}
    else:
        raise ValueError('Unknown name for config', str(name))

    if stim_dim is not None:
        FullConfig['dataset']['stim_dim'] = stim_dim
    if hidden_size is not None:
        if 'plasticnet' in FullConfig:
            FullConfig['plasticnet']['hidden_size'] = hidden_size
        elif 'tvt_memory' in FullConfig:
            FullConfig['tvt_memory']['hidden_size'] = hidden_size

    return deepcopy(FullConfig)


if __name__ == '__main__':
    from pprint import pprint
    config = SpecialPlasticRNNConfig
    pprint(config)
