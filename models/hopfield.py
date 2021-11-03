"""Classical and modern Hopfield networks."""
import os
import sys

import torch
import torch.nn as nn

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

class ClassicHopfieldLayer(nn.Module):
    def __init__( self, input_size, output_size, batch_size=1, steps=1,
                 clamp_val=1., decay_rate=0.9999, learning_rate=0.5,
                 learn_params=False, zero_in_detach=True,
                 take_sign_in_output=True):
        super().__init__()
        self.steps = steps
        self.input_size = input_size
        self.zero_in_detach = zero_in_detach
        self.clamp_val = clamp_val
        self.take_sign_in_output = take_sign_in_output

        if output_size is None:
            self.output_size = input_size
        else:
            self.output_size = output_size

        if self.input_size != self.output_size:
            self.diff_in_out_dim = True
        else:
            self.diff_in_out_dim = False

        self.batch_size = batch_size
        self.hebb_shape = (batch_size, self.output_size, self.input_size)
        self.register_buffer('hebb', torch.zeros(self.hebb_shape))

        if learn_params:
            self.lamb = nn.Parameter(torch.tensor(decay_rate))
            self.eta = nn.Parameter(torch.tensor(learning_rate))
        else:
            self.lamb = decay_rate
            self.eta = learning_rate

    def detach_hebb(self):
        if self.zero_in_detach:
            nn.init.zeros_(self.hebb)
        self.hebb.detach_()

    def reset_hebb(self):
        """Reset hebb to zero weights."""
        nn.init.zeros_(self.hebb)

    def update_hebb(self, pre, post, third_factor=None):
        """Compute Hebbian updates.

        Update Hebb with function of pre and post

        Args:
            pre: (batch_size, in_features)
            post: (batch_size, out_features)
            third_factor: (batch_size, 1) or (batch_size, out_features, in_features)
        """
        batch_size = pre.shape[0]
        deltahebb = torch.bmm(post.view(batch_size, self.output_size, 1),
                              pre.view(batch_size, 1, self.input_size))

        self.hebb = self.hebb * self.lamb
        if self.batch_size > 1:
            self.hebb = self.hebb + self.eta*third_factor.squeeze()[:,None,None]*deltahebb
        else:
            self.hebb = self.hebb + self.eta*third_factor*deltahebb
        self.hebb = torch.clamp(self.hebb, min=-self.clamp_val,
                                max=self.clamp_val)

    def forward(self, input):
        """
        Args:
             input: tensor (batch_size, input_features)
        Return:
            output: tensor (batch_size, output_features)
        """
        batch_size = input.shape[0]
        output = input.view(batch_size, self.input_size, 1)
        output_old = torch.zeros_like(output)
        step = 0

        while step < self.steps:
            if not self.diff_in_out_dim and (output == output_old).all():
                break # This condition is only for square HEBB

            output_old = output
            output = torch.matmul(self.hebb, output_old)
            if self.take_sign_in_output:
                output = torch.sign(output)
            step += 1
        return output.squeeze(dim=2)


class ClassicHopfield(nn.Module):
    def __init__(self, input_size, output_size=None, config=None,
                 recalc_output_in_forward=False):
        super().__init__()

        self.layer = ClassicHopfieldLayer(input_size=input_size,
                                          output_size=output_size, **config)
        self.recalc_output_in_forward = recalc_output_in_forward


    def forward(self, input, modu_input=None, rnn_state=None, target=None):
        """
        Args:
             input: tensor (seq_len, batch_size, input_size)
             modu_input: tensor (seq_len, batch_size, modu_input_size)
                Modulatory input.
        """
        seq_len, batch_size, _ = input.shape

        output = []
        steps = range(input.size(0))
        for i in steps:
            out = self.layer(input[i])
            if target is not None:
                post = target[i]
            else:
                post = input[i]
            self.layer.update_hebb(pre=input[i],
                                   post=post,
                                   third_factor=modu_input[i])
            if self.recalc_output_in_forward:
                out = self.layer(input[i])
            output.append(out)

        # output (seq_len, batch_size, input_size)
        output = torch.stack(output, 0)
        return output, None

    def reset_weights(self, batch):
        nn.init.zeros_(self.layer.hebb[batch])

