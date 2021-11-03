"""Utility for models."""

from pathlib import Path

import torch
import torch.nn as nn

from models.plasticnet import SpecialPlasticRNN
from models.hopfield import ClassicHopfield
from models.controller import Controller
from models.mlmemory import Memory as TVTMemory
import models.module as module
import tools

class MemoryNet(module.Module):
    def __init__(self, input_size, output_size, config=None):
        super(MemoryNet, self).__init__()

        hidden_size = config['hidden_size']
        self.hidden_size = hidden_size
        self.output_size = output_size
        network = config['network']
        # TODO: separate input size and modu_input size
        if network == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size)
        elif network == 'special_plastic':
            if output_size == input_size-1:
                self.rnn = SpecialPlasticRNN(input_size, **config)
            else:
                self.rnn = SpecialPlasticRNN(
                    input_size, output_size=output_size, **config
                    )
        elif network == 'tvt':
            memory_word_size = input_size - 1
            R = 40
            num_read_heads = 1
            top_k = 0
            B = 1
            self.rnn = TVTMemory(
                input_size-1,
                batch_size=B, memory_word_size=memory_word_size,
                num_rows_memory=R, num_read_heads=num_read_heads, top_k=top_k)
        elif network == 'hopfield':
            #raise NotImplementedError
            # TODO: again specialized for Recall, fix!
            self.rnn = ClassicHopfield(
                input_size=input_size-1, output_size=output_size,
                config=config['hopfield_config']
                )
        else:
            raise ValueError('Unknown network', str(network))

        self.network = network
        self.rnn_state = None

        if self.network == 'special_plastic':
            self.fc = nn.Identity()
        elif self.network == 'tvt':
            self.fc = nn.Identity()
        elif self.network == 'hopfield':
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(hidden_size, output_size)

        self.thinking_step = config['thinking_step']

        self.residual_conn = config.get('residual_conn', True)
        if network == 'tvt':
            self.residual_conn = False

    def forward(self, input, modu_input=None, target=None):
        """

        Args:
             input: (seq_len, batch_size, input_size)
             modu_input: optional, (seq_len, batch_size, modu_input_size)

        Return:
            out: (seq_len, batch_size, output_size)
            rnn_out: (seq_len, batch_size, rnn_output_size)
        """
        # Init hebb for all plastic layers
        # TODO: FIX THIS
        for mod in self.modules():
            try:
                mod.detach_hebb()
            except AttributeError:
                pass

        if self.thinking_step > 1:
            # Repeat along seq_len dimension
            x_ = torch.repeat_interleave(input, self.thinking_step, dim=0)
            if modu_input is not None:
                modu_input = torch.repeat_interleave(
                    modu_input, self.thinking_step, dim=0)
        else:
            x_ = input

        # TODO: make this more general
        if modu_input is None:
            rnn_out, self.rnn_state = self.rnn(x_, target=target)
        else:
            try:
                rnn_out, self.rnn_state = self.rnn(x_, modu_input=modu_input,
                                                   target=target)
            except TypeError:
                x_ = torch.cat((x_, modu_input), dim=-1)
                rnn_out, self.rnn_state = self.rnn(x_)

        if self.thinking_step > 1:
            rnn_out = rnn_out[::self.thinking_step, :, :]  # trim here

        out = self.fc(rnn_out)

        # Residual connections
        if self.residual_conn:
            out += input

        if self.analyzing:
            self.writer['rnn_out'].append(rnn_out)

        return out, rnn_out

    def load(self, f, map_location=None, load_hebb=True, **kwargs):
        state_dict = torch.load(f, map_location=map_location)
        if load_hebb:
            self.load_state_dict(state_dict)
        else:
            keys = list(state_dict.keys())
            for key in keys:
                if key[-5:] == '.hebb':
                    print('Not loading ', key)
                    del state_dict[key]
            self.load_state_dict(state_dict, strict=False)


class ControllerModel(module.Module):
    """ Wrapper around CtrlPlasticRNN """

    def __init__(
            self, input_size, output_size, ctrl_config, mem_config,
            ):
        super(ControllerModel, self).__init__()
        self.input_size = input_size
        self.ctrl_size = ctrl_config['ctrl_size']
        self.memory_size = mem_config['hidden_size']
        self.output_size = output_size
        self.ctrl_config = ctrl_config
        self.mem_config = mem_config

        network = mem_config['network']
        self.network = network
        if network == 'special_plastic':
            self.mem_net = SpecialPlasticRNN(self.ctrl_size+1, **mem_config)
        elif network == 'tvt':
            raise ValueError('Currently not implemented')
            memory_word_size = input_size - 1
            R = 5
            num_read_heads = 1
            top_k = 0
            B = 1
            self.mem_net = TVTMemory(
                input_size-1,
                batch_size=B, memory_word_size=memory_word_size,
                num_rows_memory=R, num_read_heads=num_read_heads, top_k=top_k)
        elif network == 'hopfield':
            self.mem_net = ClassicHopfield(
                input_size=self.ctrl_size,
                config=mem_config['hopfield_config'],
                recalc_output_in_forward=True
                )
        else:
            raise ValueError(f'Unknown network {network}')

        if "train_memnet" in ctrl_config.keys() and not ctrl_config["train_memnet"]:
            print("Turning off grad in memory network")
            for p in self.mem_net.parameters():
                p.requires_grad = False

        if ctrl_config['ctrl_type'] is not None:
            self.use_ctrl = True
            self.ctrl_net = Controller(input_size, self.ctrl_size, **ctrl_config)
            if self.ctrl_size == output_size:
                self.fc = nn.Identity()
            else:
                self.fc = nn.Linear(self.ctrl_size, output_size)
        else:
            self.use_ctrl = False

        self.learn_modu = ctrl_config['learn_modu']

        self.init_ctrl_params()
        if ctrl_config['activ'] == 'identity':
            self.fc_activ = nn.Identity()
        else:
            self.fc_activ = nn.Tanh()

    def forward(self, input, modu_input=None):
        # input size (seq_len, batch_size, input_size)
        # output size (seq_len, batch_size, output_size)

        # Init hebb for all
        for mod in self.modules():
            try:
                mod.detach_hebb()
            except AttributeError:
                pass
        prev_out = self.reinit_ctrl_params()
        seq_len, batch_size, input_size = input.shape
        output = []

        for i in range(seq_len):
            current_input = input[i]
            if not self.learn_modu:
                current_modu_in = torch.unsqueeze(modu_input[i], 0)
            else:
                current_modu_in = None

            if self.use_ctrl:
                new_in = torch.unsqueeze(current_input, 0)
                mem_in, mem_target, modu_input_i = self.ctrl_net(
                    new_in, prev_out, current_modu_in
                )
                mem_out, _ = self.mem_net(
                    mem_in, modu_input=modu_input_i, target=mem_target
                )
                fc_out = self.fc(mem_out)
                output.append(self.fc_activ(fc_out))
                prev_out = (mem_in, mem_out, modu_input_i)
            else:
                new_in = torch.unsqueeze(current_input, 0)
                mem_out,_ = self.mem_net(
                    torch.sign(prev_out),
                    modu_input=current_modu_in,
                    target=new_in
                )
                mem_out = self.fc_activ(mem_out)
                output.append(mem_out)
                prev_out = mem_out
        output = torch.stack(output, 0)
        if self.analyzing:
            self.writer['rnn_out'].append(output)
        return output, output

    def init_ctrl_params(self):
        if self.use_ctrl:
            self.prev_mem_in_bias = nn.Parameter(
                torch.randn(1, 1, self.ctrl_size)* 0.05
            )
            self.prev_mem_out_bias = nn.Parameter(
                torch.randn(1, 1, self.ctrl_size)* 0.05
            )
            self.prev_modu_input_bias = nn.Parameter(torch.randn(1,1,1))
        else:
            self.prev_mem_in_bias = nn.Parameter(
                torch.randn(1, 1, self.ctrl_size)* 0.05
            )

    def reinit_ctrl_params(self):
        if self.use_ctrl:
            init_states = (
                self.prev_mem_in_bias.clone(),
                self.prev_mem_out_bias.clone(),
                self.prev_modu_input_bias.clone()
            )
            return init_states
        else:
            return self.prev_mem_in_bias.clone()


def get_model_from_config(config, input_size=None, output_size=None):
    input_size, output_size = _try_infer_input_output_size(
        config, input_size, output_size)

    if 'ctrlnet' in config:
        net = ControllerModel(
            input_size, output_size, ctrl_config=config['ctrlnet'],
            mem_config=config['plasticnet']
            )
    elif 'tem' in config:
        if 'plasticnet' in config:
            net = TemPlasticNet(config['tem'], config)
        else:
            net = TemModel(config['tem'], config)
    elif 'plasticnet' in config:
        net = MemoryNet(input_size, output_size, config=config['plasticnet'])
    elif 'hopfield' in config:
        # TODO: This hidden size smaller than input size is a special temporary
        # setting, need removal soon
        net = ClassicHopfield(
            input_size=input_size-1, output_size=output_size,
            config=config['hopfield']
            )
    elif 'tvt_memory' in config:
        net = MemoryNet(input_size, output_size,config=config['tvt_memory'])

    else:
        # TODO: Implement other types of models here
        raise NotImplementedError('Unimplemented config type' +  str(config))
    return net


def _try_infer_input_output_size(config, input_size=None, output_size=None):
    if input_size is None:
        if 'input_size' in config:
            input_size = config['input_size']
        else:
            try:
                if config['dataset']['name'] == 'recall':
                    input_size = config['dataset']['stim_dim'] + 1
            except AttributeError:
                raise ValueError('input_size cannot be inferred')

    if output_size is None:
        if 'output_size' in config:
            output_size = config['output_size']
        else:
            try:
                if config['dataset']['name'] == 'recall':
                    output_size = config['dataset']['stim_dim']
            except AttributeError:
                raise ValueError('output_size cannot be inferred')

    return input_size, output_size


def get_model(config_or_path):
    """Get model from config or path.

    This should be the only public interface of models.

    Args:
        config_or_path: config file or path to model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(config_or_path, dict):
        # is config
        config = config_or_path
        net = get_model_from_config(config)
    elif isinstance(config_or_path, str):
        if not tools.islikemodeldir(config_or_path):
            raise FileNotFoundError('No model in ', config_or_path)
        config = tools.load_config(config_or_path)
        net = get_model_from_config(config)
        # Convention, make more general
        model_path = Path(config_or_path) / 'model.pt'
        if model_path.is_file():
            net.load(model_path, map_location=device)
        else:
            raise FileNotFoundError('Could not find {}'.format(model_path))
    net = net.to(device)
    return net
