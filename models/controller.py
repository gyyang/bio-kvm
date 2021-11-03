import torch
import torch.nn as nn
import models.module as module
from math import sqrt

import numpy as np

EPSILON = 1e-10 #to avoid divide-by-zero

class Controller(module.Module):

    def __init__(self, input_size, output_size,
                 ctrl_type=None, learn_modu=False, complicated_modu=False,
                 normalize=True,
                 **kwargs
                 ):

        super(Controller, self).__init__()
        self.input_size = input_size

        if ctrl_type is None:
            self.full_input_size = input_size
        else:
            self.full_input_size = input_size + output_size*2 + 2

        if learn_modu:
            self.full_input_size -= 1
        self.in_ctrl_size = self.full_input_size #output_size                                         
        self.out_ctrl_size = self.full_input_size
        self.output_size = output_size
        self.norm_val = sqrt(output_size)
        self.ctrl_type = ctrl_type
        self.learn_modu = learn_modu
        self.complicated_modu = complicated_modu
        self.normalize = normalize

        # Controller
        if ctrl_type == "LSTM":
            self.ctrl2memin = nn.LSTM(
                self.in_ctrl_size, self.output_size, num_layers=1
            )
            self.ctrl2memout = nn.LSTM(
                self.out_ctrl_size, self.output_size, num_layers=1
            )
        elif 'FF' in ctrl_type:
            self.ctrl2memin = nn.Sequential(                                    
                nn.Linear(self.in_ctrl_size, self.output_size),
                nn.Tanh()
            )                                                                   
            self.ctrl2memout = nn.Sequential(                                   
                nn.Linear(self.out_ctrl_size, self.output_size),
                nn.Tanh()
            )   
        elif ctrl_type is None:
            assert(input_size == output_size)
            self.ctrl2memin = nn.Identity()
            self.ctrl2memout = nn.Identity()
        else:
            raise ValueError(f"Invalid controller net specification {ctrl_type}")

        # Learn Modulatory Input
        if learn_modu:
            if complicated_modu:
                if ctrl_type == 'LSTM':
                    self.modu_net = nn.LSTM(self.in_ctrl_size, 1, num_layers=1)
                elif ctrl_type == 'FF':
                    self.modu_net = nn.Sequential(
                        nn.Linear(self.in_ctrl_size, 1),
                        Clamp(floor=0, ceil=1)
                        )
                elif ctrl_type == 'FFSigmoid':
                    self.modu_net = nn.Sequential(
                        nn.Linear(self.in_ctrl_size, 1),
                        nn.Sigmoid()
                        )
                elif ctrl_type is None:
                    self.modu_net = nn.Sequential(
                        nn.Linear(self.input_size, 1),
                        nn.ReLU()
                    )
            else:
                self.modu_net = nn.Sequential(
                    nn.Linear(self.input_size, 1), nn.ReLU()
                )

    def forward(self, input, prev_states, modu_input=None):
        """
        Args:
             input: tensor (seq_len, batch_size, input_size)
             modu_input: tensor (seq_len, batch_size, modu_input_size)
                Modulatory input.
        """

        seq_len, batch_size, _ = input.shape
        prev_mem_in, prev_mem_out, prev_modu_input = prev_states
        mem_in = []
        mem_target = []
        all_modu_input = []
        steps = range(input.size(0))
        for i in steps:
            current_input = torch.unsqueeze(input[i], 0)
            if self.learn_modu:
                stacked_input = torch.cat(
                    [current_input, prev_mem_in,
                    prev_mem_out, prev_modu_input],
                    axis=2
                )
                if self.complicated_modu:
                    modu_input_i = self.modu_net(stacked_input)
                    if self.ctrl_type == 'LSTM':
                        modu_input_i = modu_input_i[0]
                        modu_input_i = nn.functional.relu(modu_input_i)
                else:    
                    modu_input_i = self.modu_net(current_input)
            else:
                modu_input_i = torch.clone(modu_input[i])
                modu_input_i = torch.unsqueeze(modu_input_i, 0)
                stacked_input = torch.cat(                                      
                    [current_input, modu_input_i, prev_mem_in,                  
                    prev_mem_out, prev_modu_input],                             
                    axis=2                                                      
                )

            if self.ctrl_type is None:
                mem_in_i = prev_mem_out
                mem_target_i = current_input
            else:
                mem_in_i = self.ctrl2memin(stacked_input)
                mem_target_i = self.ctrl2memout(stacked_input)

            if self.ctrl_type == 'LSTM':
                mem_in_i = mem_in_i[0]
                mem_target_i = mem_target_i[0]

            if self.normalize:
                mem_in_i = self.norm_val*mem_in_i/(torch.norm(mem_in_i)+EPSILON)
                mem_target_i = self.norm_val*mem_target_i/(torch.norm(mem_target_i)+EPSILON)

            mem_in.append(mem_in_i)
            mem_target.append(mem_target_i)
            all_modu_input.append(modu_input_i)
        if len(steps) > 1:
            mem_in = torch.stack(mem_in, 0)
            mem_target = torch.stack(mem_target, 0)
            all_modu_input = torch.stack(all_modu_input, 0)
        else:
            mem_in = mem_in[0]
            mem_target = mem_target[0]
            all_modu_input = all_modu_input[0]

        return mem_in, mem_target, all_modu_input

class Clamp(nn.Module):

    def __init__(self, floor, ceil):
        super(Clamp, self).__init__()
        self.floor = floor
        self.ceil = ceil

    def forward(self, input):
        return torch.clamp(input, min=self.floor, max=self.ceil)

