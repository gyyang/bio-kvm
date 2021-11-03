# Training
import shutil
import os
import time
from itertools import chain

import numpy as np  # this seems to be useful on cluster, even if not used
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pprint import pprint

from datasets.dataset_utils import get_dataset
from models.model_utils import get_model
import tools
import configs

def train(config, reload=False):
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

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    old_loss = float('inf')
    running_loss = 0.0
    running_loss_reg = 0.0
    running_acc = 0.0
    print_every_steps = config['print_every_steps']

    time_task = 0
    time_net = 0
    loss_reg = torch.zeros(1)
    if not config['training']:
        torch.no_grad()
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
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if 'modu_input' in data.keys():
            if 'input_heteroassociative' in data.keys():
                outputs, rnn_out = net(input=data['input'],
                                       modu_input=data['modu_input'],
                                       target=data['input_heteroassociative']
                                       )
            else:
                outputs, rnn_out = net(input=data['input'],
                                       modu_input=data['modu_input'])
        else:
            if 'input_heteroassociative' in data.keys():
                outputs, rnn_out = net(
                    input=data['input'], target=data['input_heteroassociative']
                    )
            else:
                outputs, rnn_out = net(input=data['input'])
        outputs = outputs.view(-1, output_size)

        loss = criterion(outputs, data['target'])
        loss = torch.sum(loss, dim=1)
        mask_sum = torch.sum(data['mask'])
        loss_task = torch.sum(loss * data['mask']) / mask_sum
        loss = loss_task

        with torch.no_grad():
            if 'acc_binarize' in config:
                outputs = outputs > 0.5
                targets = data['target'] > 0.5
                match = (outputs == targets).float()
            else:
                outputs = torch.sign(outputs)
                match = (outputs == data['target']).float()
            acc = torch.sum(data['mask'] * torch.mean(match, dim=1)) / mask_sum

        if config['training']:
            loss.backward()
        optimizer.step()

        time_net += time.time() - start_time

        # print statistics
        running_loss += loss.item()
        running_loss_reg += loss_reg.item()
        running_acc += acc.item()

        if step % print_every_steps == 0:
            if step > 0:
                running_loss /= print_every_steps
                running_loss_reg /= print_every_steps
                running_acc /= print_every_steps

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
                '[{:5d}] loss: {:0.3f}, loss reg: {:0.3f}, acc: {:0.3f}/{:0.3f}'.format(
                    step + 1, running_loss, running_loss_reg, running_acc,
                    dataset.chance))
            print('Time per step {:0.3f}ms'.format(1e3*(time_task+time_net)/(step+1)))
            print('Total time on task {:0.3f}s, net {:0.3f}s'.format(time_task,
                                                                     time_net))

            model_path = os.path.join(config['save_path'], 'model.pt')
            torch.save(net.state_dict(), model_path)

            # optional early stop
            if running_acc > config['converged_acc_thres']:
                print('Stopping early. Accuracy ({:.4f}) above threshold ({:.4f})'
                      .format(running_acc, config['converged_acc_thres']))
                break

            loss_decrease_frac = 1 - running_loss/(old_loss + 1E5)
            if 0 < loss_decrease_frac < config['converged_loss_decrease_frac']:
                print('Stopping early. Fraction loss decrease ({:.4f}) below threshold ({:.4f})'
                      .format(loss_decrease_frac, config['converged_loss_decrease_frac']))
                break

            old_loss = running_loss
            running_loss = 0.0
            running_loss_reg = 0.0
            running_acc = 0.0

    writer.close()
    print('Finished Training\n')
    return net


def train_from_path(path):
    """Train from a path with a config file in it."""
    config = tools.load_config(path)
    return train(config)


if __name__ == '__main__':
    config = configs.get_config('emergeplasticity')

    net = train(config)
