# Training
import shutil
import os
import time
from itertools import chain
from pprint import pprint
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import cma

from datasets.dataset_utils import get_dataset
from models.model_utils import get_model
import tools


def evolve(config, reload=False):
    #TODO: This is copied from and almost identical to training.train(). Make
    #object-oriented and merge (write base class which sets up training, and
    #inherit from it to override the step() method)

    # Training networks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('Using device ', device)
    print('Training configuration:')
    pprint(config)

    dataset = get_dataset(config['dataset'])
    # TODO: This is too specialized and introduce coupling between dataset
    #  and model. Fix
    input_size = dataset.input_dim
    output_size = dataset.output_dim
    # TODO: again, this code need rethinking
    config['input_size'] = input_size
    config['output_size'] = output_size

    save_path = config['save_path']
    if not reload:
        try:
            shutil.rmtree(save_path)
        except FileNotFoundError:
            pass

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Save config
    tools.save_config(config, save_path=save_path)

    writer = SummaryWriter(save_path)

    net = get_model(config)
    net.rnn.ltf.b0.requires_grad_(False)
    net.rnn.ltf.b1.requires_grad_(False)
    criterion = nn.MSELoss(reduction='none')

    parameter_names, parameter_init = flatten_parameters(net)
    es_optimizer = cma.CMAEvolutionStrategy(parameter_init, sigma0=0.3)

    running_loss = 0.
    running_acc = 0.
    n_accumulated = 0

    time_task = 0
    time_net = 0

    for step in range(config['train_steps']):
        start_time = time.time()
        data = dataset.generate()
        for key, val in data.items():
            data[key] = torch.from_numpy(val).float().to(device)

        # Add batch dimension
        data['input'] = data['input'].unsqueeze(1)
        if 'modu_input' in data.keys():
            data['modu_input'] = data['modu_input'].unsqueeze(1)

        time_task += time.time() - start_time

        start_time = time.time()

        # generate candidate solutions
        candidate_params = es_optimizer.ask()

        # evaluate loss and acc on each candidate
        losses = []
        accs = []
        for params in candidate_params:
            #this can be parallelized via es_optimizer.optimize(obj_fn)
            set_parameters(net, parameter_names, params) #in-place

            #evaluate
            if 'modu_input' in data.keys():
                outputs, rnn_out = net(input=data['input'],
                                       modu_input=data['modu_input'])
            else:
                outputs, rnn_out = net(input=data['input'])
            outputs = outputs.view(-1, output_size)

            #loss
            loss = criterion(outputs, data['target'])
            loss = torch.sum(loss, dim=1)
            mask_sum = torch.sum(data['mask'])
            loss_task = torch.sum(loss * data['mask']) / mask_sum
            losses.append(loss_task.item())

            #acc
            if 'acc_binarize' in config:
                outputs = outputs > 0.5
                targets = data['target'] > 0.5
                match = (outputs == targets).float()
            else:
                outputs = torch.sign(outputs)
                match = (outputs == data['target']).float()
            acc = torch.sum(data['mask'] * torch.mean(match, dim=1)) / mask_sum
            accs.append(acc.item())

        #take optimization step based on losses
        es_optimizer.tell(candidate_params, losses)

        time_net += time.time() - start_time

        # print statistics TODO: can do this with es_optimzer.disp()
        running_loss += min(losses)
        running_acc += max(accs)
        n_accumulated += 1 #es_optimizer.popsize

        if step % config['print_every_steps'] == 0:
            es_optimizer.disp()
            running_loss /= n_accumulated
            running_acc /= n_accumulated
            for name, param in chain(net.named_parameters(), net.named_buffers()):
                if torch.numel(param) > 1:
                    std, mean = torch.std_mean(param)
                    writer.add_scalar(name+'_std', std, step)
                else:
                    mean = torch.mean(param)
                writer.add_scalar(name, mean, step)

            writer.add_scalar('loss_train', running_loss, step)
            writer.add_scalar('acc_train', running_acc, step)

            print('', flush=True)
            print(
                '[{:5d}] loss: {:0.3f}, acc: {:0.3f}/{:0.3f}'.format(
                    step + 1, running_loss, running_acc,
                    dataset.chance))
            print('Time per step {:0.3f}ms'.format(1e3*(time_task+time_net)/(step+1)))
            print('Total time on task {:0.3f}s, net {:0.3f}s'.format(time_task,
                                                                     time_net))

            model_path = os.path.join(config['save_path'], 'model.pt')
            torch.save(net.state_dict(), model_path)

            running_loss = 0.
            running_acc = 0.
            n_accumulated = 0

    writer.close()
    print('Finished Training\n')
    es_optimizer.result_pretty()
    return net


def flatten_parameters(net):
    parameters = []
    names = []
    for name, param in net.named_parameters():
        if param.numel() != 1:
            raise NotImplementedError("Can't flatten parameter {}: numel={}"
                                      .format(name, param))
        if param.requires_grad:
            parameters.append(param.item())
            names.append(name)
    return names, parameters


def set_parameters(net, names, flattened_params):
    with torch.no_grad():
        net = deepcopy(net)
        for name, param in zip(names, flattened_params):
            eval('net.{}.fill_(param)'.format(name)) #TODO: do this better
    return net
