# Training
import shutil
import os
from itertools import chain
from pprint import pprint

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import cma


from datasets.dataset_utils import get_dataset
from models.model_utils import get_model
import tools

torch.no_grad()

class CMAES:
    def __init__(self, config):
        self.numevals = 0
        self.running_loss = 0.0
        self.running_acc = 0.0

        self.seed = config['seed']
        self._seed_torch()
        print('Training configuration:')
        pprint(config)

        self.criterion = nn.MSELoss(reduction='none')

        self.dataset = get_dataset(config['dataset'])
        config['input_size'] = self.input_size = self.dataset.input_dim
        config['output_size'] = self.output_size = self.dataset.output_dim
        self.config = config

        self.net = get_model(config)
        self.parameter_names, self.parameter_init = flatten_parameters(self.net)

        self.save_path = config['save_path']
        reload = False
        if not reload:
            try:
                shutil.rmtree(self.save_path)
            except FileNotFoundError:
                pass

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        tools.save_config(config, save_path=self.save_path)

        self.writer = SummaryWriter(self.save_path)

    def loss(self, parameters):
        """
        Args:
            parameters (list): list of parameters optimized by CMA-ES.
        Returns:
            float: fitness i.e. loss of that parameterization on a dataset.

        """
        self._seed_torch()
        self.data = self.dataset.generate()
        for key, val in self.data.items():
            self.data[key] = torch.from_numpy(val).float()
        self.data['input'] = self.data['input'].unsqueeze(1)

        self.net = get_model(self.config)
        set_parameters(self.net, self.parameter_names, parameters)

        #forward
        if 'modu_input' in self.data.keys():
            outputs, rnn_out = self.net(input=self.data['input'],
                                   modu_input=self.data['modu_input'])
        else:
            outputs, rnn_out = self.net(input=self.data['input'])
        outputs = outputs.view(-1, self.output_size)
        loss = self.criterion(outputs, self.data['target'])
        loss = torch.sum(loss, dim=1)
        mask_sum = torch.sum(self.data['mask'])
        loss_task = torch.sum(loss * self.data['mask']) / mask_sum
        loss = loss_task

        if 'acc_binarize' in self.config:
            outputs = outputs > 0.5
            targets = self.data['target'] > 0.5
            match = (outputs == targets).float()
        else:
            outputs = torch.sign(outputs)
            match = (outputs == self.data['target']).float()
        acc = torch.sum(self.data['mask'] * torch.mean(match, dim=1)) / mask_sum

        self.running_loss += loss.item()
        self.running_acc += acc.item()

        if self.numevals % self.config['print_every_steps'] == 0:
            if self.numevals > 0:
                self.running_loss /= self.config['print_every_steps']
                self.running_acc /= self.config['print_every_steps']

                for name, param in chain(self.net.named_parameters(), self.net.named_buffers()):
                    if torch.numel(param) > 1:
                        std, mean = torch.std_mean(param)
                        self.writer.add_scalar(name+'_std', std, self.numevals)
                    else:
                        mean = torch.mean(param)
                    self.writer.add_scalar(name, mean, self.numevals)

                self.writer.add_scalar('loss_train', self.running_loss, self.numevals)
                self.writer.add_scalar('acc_train', self.running_acc, self.numevals)
            print('', flush=True)
            print(
                '[{:5d}] loss: {:0.3f}, acc: {:0.3f}/{:0.3f}'.format(
                    self.numevals + 1, self.running_loss,  self.running_acc,
                    self.dataset.chance))

            model_path = os.path.join(self.config['save_path'], 'model.pt')
            torch.save(self.net.state_dict(), model_path)

            self.running_loss = 0.0
            self.running_acc = 0.0

        self.numevals += 1
        return loss.item()

    def _seed_torch(self):
        if self.seed is not None:
            self.torchrng = torch.manual_seed(self.seed)


def evolve(config):
    cmaes = CMAES(config)

    #seed pycma
    opts = cma.CMAOptions()
    if config['seed'] is not None:
        opts.set('seed', config['seed'])
        opts['seed'] = config['seed']

    #run CMAES
    #http://cma.gforge.inria.fr/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html
    es = cma.CMAEvolutionStrategy(cmaes.parameter_init, 0.3, opts)
    es.optimize(cmaes.loss, iterations=config['train_steps']) #TODO: this is wrong

    es.result_pretty()
    # es.ask()
    # es.logger.plot()

    cmaes.writer.close()

    final_params = es.result.xfavorite
    set_parameters(cmaes.net, cmaes.parameter_names, final_params)
    return cmaes.net


#############
## Helpers ##
#############
def flatten_parameters(net):
    parameters = []
    names = []
    for name, param in net.named_parameters():
        if param.numel() != 1:
            raise NotImplementedError("Can't flatten parameter {}: numel={}"
                                      .format(name, param.numel()))
        if param.requires_grad:
            parameters.append(param.item())
            names.append(name)
    return names, parameters


def set_parameters(net, names, flattened_params):
    with torch.no_grad():
        for name, param in zip(names, flattened_params):
            eval('net.{}.fill_(param)'.format(name)) #TODO: do this better
