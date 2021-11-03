# Model

import math
import warnings

import torch
import torch.nn as nn

import models.module as module

import matplotlib.pyplot as plt
import numpy as np

EPSILON = 1e-10 #to avoid divide-by-zero

####################
# Plastic networks #
####################

class SpecialPlasticRNN(module.Module):
    """Specialized mainly feedforward network with plasticity.

    Input projects to hidden which projects output. Input-to-hidden and
    hidden-to-output are subject to different trainable plasticity rules.

    Args:
        input_size: int
        hidden_size: int
        i2h: dict, config of i2h layer
        h2o: dict, config of h2o layer
        use_global_thirdfactor: bool
        local_thirdfactor: dict, config of local thirdfactor
        hebb: dictionary of hebbian parameters
        predictive_gate: bool, if True use predictive gate
    """

    def __init__(self, input_size, hidden_size, i2h, h2o,
                 local_thirdfactor = None,
                 h2o_use_local_thirdfactor=False,
                 use_global_thirdfactor=False,
                 predictive_gate=False,
                 reset_to_reference=False,
                 recalc_output_in_forward=False,
                 output_size=None,
                 batch_size=1,
                 softmax_temp=1,
                 **kwargs
                 ):
        """Specialized Plastic RNN."""
        super(SpecialPlasticRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.activ = torch.nn.Tanh()

        self.ltf = LocalThirdFactor(hidden_size, batch_size, **local_thirdfactor)
        self.i2h = PlasticLinear(input_size-1, hidden_size, batch_size, **i2h)
        if output_size is None:
            self.h2o = PlasticLinear(hidden_size, input_size-1, batch_size, **h2o)
        else:
            self.h2o = PlasticLinear(hidden_size, output_size, batch_size, **h2o)

        # self.activation = torch.tanh
        self.activation = nn.Identity()
        self.softmax = nn.Softmax(dim=1)

        # input to third factor
        self.i2tf = nn.Linear(1, 1)
        nn.init.ones_(self.i2tf.weight)
        nn.init.zeros_(self.i2tf.bias)
        self.use_global_thirdfactor = use_global_thirdfactor
        self.predictive_gate = predictive_gate
        self.softmax_temp = softmax_temp
        if self.predictive_gate == 'sigmoid':
            #offset and steepness chosen so that:
            #scaled_match = -10 if match_avg==1 (perfect match)
            #             = +10 if match_avg==0.5 (random)
            self.pred_gate_offset = nn.Parameter(torch.tensor(0.75))
            self.pred_gate_steepness = nn.Parameter(torch.tensor(40.))

        self.h2o_use_local_thirdfactor = h2o_use_local_thirdfactor
        if h2o['decay'] == 'active' and not h2o_use_local_thirdfactor:
            print('Warning: h2o needs access to local_thirdfactor for "active" decay')

        self.recalc_output_in_forward = recalc_output_in_forward

        if reset_to_reference:
            self.reset_to_reference()


    def forward(
        self, input, modu_input=None, rnn_state=None, target=None,
        do_update=True, return_hidden_activity=False
        ):
        """
        Args:
             input: tensor (seq_len, batch_size, input_size)
             modu_input: tensor (seq_len, batch_size, modu_input_size)
                Modulatory input.

        Returns:
            output: tensor (seq_len, batch_size, hidden_size)
            rnn_state: None
        """

        seq_len, batch_size, _ = input.shape

        output = []
        h_activity = []
        softmax_h_activity = []
        steps = range(input.size(0))
        for i in steps:
            current_input = input[i]
            # Forward passes
            # Excluding the store signal input
            h_ = self.activation(self.i2h(current_input))
            if return_hidden_activity:
                h_activity.append(h_.detach().numpy())
            h = self.softmax(h_*self.softmax_temp)
            out = self.h2o(h)

            if not do_update:
                output.append(out)
                softmax_h_activity.append(h.detach().numpy())
                continue

            # Global third factor
            if self.use_global_thirdfactor:
                assert modu_input is not None, 'modu input must be provided ' \
                                               'when using global third factor'
                global_thirdfactor = self.i2tf(modu_input[i])
            else:
                global_thirdfactor = torch.ones(1).to(input.device)

            # If compute prediction error and use for gating
            # Compute prediction error, assuming balanced recall (inputs +1/-1)
            if self.predictive_gate:
                match = (torch.sign(out) == torch.sign(current_input)).float()
                match_avg = torch.mean(match)
                if self.predictive_gate == 'sigmoid':
                    gate = torch.sigmoid(self.pred_gate_steepness*(self.pred_gate_offset-match_avg))
                else:
                    gate = torch.clamp(2*(1-match_avg), min=0, max=1)
                global_thirdfactor = global_thirdfactor * gate

            # Local third factor
            local_thirdfactor = self.ltf(h, global_thirdfactor) # GTF for batching seq. LTF

            # Update Hebb passes
            i2h_thirdfactor = local_thirdfactor * global_thirdfactor
            self.i2h.update_hebb(pre=current_input, post=h,
                                 third_factor=i2h_thirdfactor)

            # Recomputing h after i2h weight is updated
            h_new_ = self.activation(self.i2h(current_input))
            h_new = self.softmax(h_new_*self.softmax_temp)

            h2o_target = current_input if target is None else target[i]
            h2o_thirdfactor = global_thirdfactor
            if self.h2o_use_local_thirdfactor:
                h2o_thirdfactor = h2o_thirdfactor * local_thirdfactor

            self.h2o.update_hebb(pre=h_new, post=h2o_target,
                                 third_factor=h2o_thirdfactor,
                                 thirdfactor_syn='pre')

            if self.recalc_output_in_forward:
                new_out = self.h2o(h_new)
                output.append(new_out)
                if return_hidden_activity:
                    softmax_h_activity.append(h_new.detach().numpy())
            else:
                output.append(out)
                if return_hidden_activity:
                    softmax_h_activity.append(h.detach().numpy())

            if self.analyzing:
                self.writer['h_presm'].append(h_)
                self.writer['h'].append(h)
                self.writer['local_thirdfactor'].append(local_thirdfactor)
                self.writer['global_thirdfactor'].append(global_thirdfactor)
                if self.ltf.mode == 'least_recent_random':
                    self.writer['prob_plastic'].append(self.ltf.prob_plastic)


        # output (seq_len, batch_size, hidden_size)
        output = torch.stack(output, 0)
        if return_hidden_activity:
            return output, None, h_activity, softmax_h_activity
        else:
            return output, None

    def reset_to_reference(self):
        """Reset weights to reference networks."""
        assert self.i2h.hebb_mode == 'affine'
        assert self.h2o.hebb_mode == 'affine'
        nn.init.constant_(self.i2tf.weight, 1.)
        nn.init.constant_(self.i2tf.bias, 0.)
        # self.i2tf = Heaviside(-0.5)  # makes no difference in standard recall

        if self.i2h.weight is not None:
            nn.init.zeros_(self.i2h.weight)
        if self.h2o.weight is not None:
            nn.init.zeros_(self.h2o.weight)
        nn.init.ones_(self.i2h.hebb_update_rate)
        nn.init.ones_(self.h2o.hebb_update_rate)
        if torch.is_tensor(self.i2h.hebb_strength):
            nn.init.ones_(self.i2h.hebb_strength)
        if torch.is_tensor(self.h2o.hebb_strength):
            nn.init.ones_(self.h2o.hebb_strength)

        if self.h2o.decay == 'passive_global':
            #this is arbitrary, should ideally depend on hidden size
            nn.init.constant_(self.h2o.decay_rate, 0.95)

        # Pre-Dependent plasticity rules
        nn.init.constant_(self.i2h.pre_fn.weight, 1.)
        nn.init.constant_(self.i2h.pre_fn.bias, 0.)
        nn.init.constant_(self.i2h.post_fn.weight, 0.)
        nn.init.constant_(self.i2h.post_fn.bias, 1.)

        # Regular Hebbian
        nn.init.constant_(self.h2o.pre_fn.weight, 1.)
        nn.init.constant_(self.h2o.pre_fn.bias, 0.)
        nn.init.constant_(self.h2o.post_fn.weight, 1.)
        nn.init.constant_(self.h2o.post_fn.bias, 0.)

    def reset_weights(self, batch=None):
        self.i2h.reset_hebb(batch=batch)
        self.h2o.reset_hebb(batch=batch)

    def detach_hebb(self):
        if 'random' in self.ltf.mode:
            self.ltf.prob_plastic.detach_()

###########
# Helpers #
###########

class Heaviside(nn.Module):
    def __init__(self, bias=0.):
        super().__init__()
        self.bias = torch.tensor(bias)

    def forward(self, input):
        return torch.heaviside(input + self.bias, torch.tensor(0.5))


class ResMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(ResMLP, self).__init__()
        # start and end with 1
        hidden_sizes = [input_size] + hidden_sizes + [input_size]
        n_layers = len(hidden_sizes)
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if i < n_layers - 2:
                layers.append(nn.ReLU())

        self.layers = layers

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        return output + input


class LocalThirdFactor(nn.Module):
    #TODO: this is better than before but the Right way to do this would be to
    #make an ABC and then have each mode be a subclass.
    def __init__(self, hidden_size, batch_size, mode='sequential', burnin=1,
                 random_scaling=1., a0=1.01, b0=0., a1=0.99, b1=0.,
                 use_modu_input=False):
        super().__init__()
        self.hidden_size = hidden_size
        if batch_size > 1 and mode != 'sequential':
            raise ValueError("Batching not implemented for non-sequential TFs")
        self.batch_size = batch_size
        self.random_scaling = random_scaling
        self.use_modu_input = use_modu_input

        _valid_modes = ['sequential', 'least_recent', 'random',
                       'random_least_recent', 'random_linear_dynamics']
        if mode in _valid_modes:
            self.mode = mode
        else:
            raise ValueError('Invalid value for local thirdfactor mode: {}'
                             .format(mode))


        if mode == 'random':
            self.a0 = 0.
            self.a1 = 0.
            if b0 == 0:
                self.b0 = self.b1 = nn.Parameter(torch.tensor(1./hidden_size))
            else:
                self.b0 = self.b1 = nn.Parameter(torch.tensor(b0))
        elif mode == 'random_least_recent':
            self.a0 = nn.Parameter(torch.tensor(a0))
            self.b0 = nn.Parameter(torch.tensor(b0))
            self.a1 = 0.
            self.b1 = 0.
        elif mode == 'random_linear_dynamics':
            self.a0 = nn.Parameter(torch.tensor(a0))
            self.b0 = nn.Parameter(torch.tensor(b0))
            self.a1 = nn.Parameter(torch.tensor(a1))
            self.b1 = nn.Parameter(torch.tensor(b1))
        self.reset(burnin)


    def reset(self, burnin=1000):
        if self.mode == 'sequential':
            self.slot_i = torch.zeros(self.batch_size)
        elif self.mode == 'least_recent':
            #like sequential but in random order
            self.time_since_plastic = torch.rand(1, self.hidden_size)
        elif self.mode.startswith('random'):
            #TODO: allow passing in init scalar or vector
            self.prob_plastic = self.random_scaling*torch.ones(1, self.hidden_size)/self.hidden_size
            for t in range(burnin):
                #let dynamics of prob_plastic reach steady state
                dummy_hidden_activation = torch.zeros(1,self.hidden_size)
                local_thirdfactor = self(dummy_hidden_activation)
                self.update_prob_plastic(local_thirdfactor)


    def update_prob_plastic(self, local_thirdfactor):
        self.prob_plastic = torch.clamp(
                (self.a0*self.prob_plastic+self.b0)*(1-local_thirdfactor)
                + (self.a1*self.prob_plastic+self.b1)*local_thirdfactor,
            min=0., max=1.)


    def forward(self, hidden_activation, modu_input=None):
        if self.mode is None:
            local_thirdfactor = 1.
        elif self.mode == 'sequential':
            local_thirdfactor = torch.zeros_like(hidden_activation)
            if modu_input is None or not self.use_modu_input:
                on_or_off = torch.ones_like(self.slot_i)
            else:
                on_or_off = (modu_input[:,0] > 0).float()
            local_thirdfactor.index_put_(
                (torch.arange(self.batch_size).long(), self.slot_i.long()),
                on_or_off)
            self.slot_i = torch.fmod(self.slot_i + on_or_off, self.hidden_size)
        elif self.mode == 'least_recent':
            local_thirdfactor = torch.zeros_like(hidden_activation)
            least_recent_idx = torch.argmax(self.time_since_plastic) #TODO: needs batching?
            local_thirdfactor[:, least_recent_idx] = 1.
            self.time_since_plastic[:, least_recent_idx] = 0
        elif self.mode.startswith('random'):
            local_thirdfactor = (torch.rand_like(hidden_activation) <
                                 self.prob_plastic).float()
            self.update_prob_plastic(local_thirdfactor)
        return local_thirdfactor


class PlasticLinear(module.Module):
    r"""Applies a plastic linear transformation to the incoming data: :math:`y = xA^T + b`

    When Hebb is not updated, this layer should behave like a linear layer (but slower)

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        weight: If set to ``True``, the layer will contain a trainable weight
            Default: ``True``
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        mode: str, Mode of Hebbian update, 'pointwise', 'affine', 'mlp'
            Describes how pre and post values are combined to compute the
            update
        decay: str or None, 'active' (decay_rate = 1-update_rate),
                    'passive_global' (one decay_rate for all synapses),
                    'passive_neuron' (each neuron has decay_rate),
                    'passive_synapse' (each synapse has decay_rate),
                    or None (no decay)
        clamp: positive float
            Hebb plasticity clamped between -clamp and +clamp
        init_affine_weight: float, init weight of affine # TODO: make more general
        init_affine_bias: float, init bias of affine mode
        hebb_strength: str, 'tensor' or 'scalar', whether trainable
            hebb_strength is tensor or scalar
        stability: bool, if True, introduce a stability tensor same shape as weight
            this tensor describes how stable each weight should be

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape (out_features, in_features)
        bias:   the learnable bias of the module of shape (out_features,)
        hebb:   hebbian weights of the module of shape (batch_size, out_features, in_features)

    """
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self,
                 in_features,
                 out_features,
                 batch_size,
                 weight=True,
                 bias=True,
                 mode=None,
                 decay=None,
                 clamp=1.,
                 hebb_strength='tensor',
                 stability=False,
                 normalize=False,
                 normalize_scale=1.,
                 init_affine_weight=1.,
                 init_affine_bias=0.,
                 init_hebb_update_rate=0.5,
                 init_hebb_strength=0.5,
                 zero_in_detach=False,
                 stability_threshold=10
                 ):
        super(PlasticLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if weight:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.register_parameter('weight', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if hebb_strength == 'tensor':
            self.hebb_strength = nn.Parameter(
                init_hebb_strength * torch.randn(out_features, in_features))
        elif hebb_strength == 'scalar':
            self.hebb_strength = nn.Parameter(
                torch.ones(1,) * init_hebb_strength)
        elif hebb_strength is None:
            self.hebb_strength = 1.
        else:
            raise ValueError('Unknown hebb_strength ', hebb_strength)

        self.normalize = normalize
        if normalize == 'column':
            self.normalize_dim = 1 #self.hebb.shape=[n_batch, out_feat, in_feat]
        elif normalize == 'row':
            self.normalize_dim = 2
        elif normalize is not False:
            raise ValueError('Unknown weight normalization mode: {}'.format(normalize))
        if normalize is not False:
            self.normalize_scale = nn.Parameter(torch.tensor(normalize_scale))


        # Trainable scalar update rate
        self.hebb_update_rate = nn.Parameter(
            init_hebb_update_rate * torch.ones(1))
        self.hebb_mode = mode

        if decay == 'active':
            #not a parameter: recalculated at every call of update_hebb()
            self.decay_rate = None
        elif decay == 'passive_global':
            self.decay_rate = nn.Parameter(torch.rand(1)/10+0.9)
        elif decay == 'passive_neuron':
            self.decay_rate = nn.Parameter(torch.rand(in_features)/10+0.9)
        elif decay == 'passive_synapse':
            self.decay_rate = nn.Parameter(torch.rand(out_features, in_features)/10+0.9)
        elif decay is None:
            self.decay_rate = torch.tensor(1.)
        else:
            raise ValueError('Invalid value for decay: {}'.format(decay))
        self.decay = decay
        self.hebb_clamp = clamp

        if self.hebb_mode == 'pointwise':
            # Residual MLP
            self.pre_fn = ResMLP(input_size=1, hidden_sizes=[5, 5])
            self.post_fn = ResMLP(input_size=1, hidden_sizes=[5, 5])
        elif self.hebb_mode == 'affine':
            self.pre_fn = nn.Linear(in_features=1, out_features=1)
            self.post_fn = nn.Linear(in_features=1, out_features=1)
            nn.init.constant_(self.pre_fn.weight, init_affine_weight)
            nn.init.constant_(self.post_fn.weight, init_affine_weight)
            nn.init.constant_(self.pre_fn.bias, init_affine_bias)
            nn.init.constant_(self.post_fn.bias, init_affine_bias)
        elif self.hebb_mode == 'mlp':
            self.hebb_fn = nn.Sequential(
                nn.Linear(2, 5),
                nn.ReLU(),
                nn.Linear(5, 5),
                nn.ReLU(),
                nn.Linear(5, 1)
            )

        self.batch_size = batch_size
        self.hebb_shape = (batch_size, self.out_features, self.in_features)
        self.register_buffer('hebb', torch.zeros(self.hebb_shape))

        self.stability = stability
        if stability:
            self.register_buffer('hebb_stability',
                                 torch.zeros(self.hebb_shape))
        self.reset_parameters()
        self.zero_in_detach = zero_in_detach
        self.stability_threshold = stability_threshold

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def detach_hebb(self):
        """Detach hebb from graph."""
        self.hebb.detach_()
        if self.zero_in_detach:
            nn.init.zeros_(self.hebb)
            if self.stability:
                nn.init.zeros_(self.hebb_stability)

    def reset_hebb(self, batch=None):
        """Reset hebb to zero weights. Minimal noise to break ties when taking
        torch.sign(output) for accuracy evaluation."""
        if batch is None:
            nn.init.uniform_(self.hebb, a=-EPSILON, b=EPSILON)
        else:
            reset_val = torch.zeros_like(self.hebb[batch])
            nn.init.uniform_(reset_val, a=-EPSILON, b=EPSILON)
            self.hebb[batch,:,:] = reset_val
        if self.stability:
            nn.init.zeros_(self.hebb_stability)

    def update_hebb(self, pre, post, third_factor=None,
                    thirdfactor_syn='post'):
        """Compute Hebbian updates.

        Update Hebb with function of pre and post

        Args:
            pre: (batch_size, in_features)
            post: (batch_size, out_features)
            third_factor: third factor
                shape (batch_size, 1) or (batch_size, out_features)
                (batch_size, out_features, in_features)
            thirdfactor_syn: str, 'pre' or 'post', whether third factors are
                pre-synaptic or post-synaptic. Only used when thirdfactor is
                (batch_size, in_features or out_features)

        """
        batch_size = pre.shape[0]
        if self.hebb_mode in ['pointwise', 'affine']:
            pre = self.pre_fn(pre.view(-1, 1))
            post = self.post_fn(post.view(-1, 1))

        # Batched outer product, deltahebb (batch_size, out_features, in_features)
        # deltahebb = torch.bmm(post.view(batch_size, self.out_features, 1),
        #                       pre.view(batch_size, 1, self.in_features))

        # The same could be achieved with batch outer product torch.bmm, but slightly slower
        post_ = post.view(batch_size, self.out_features, 1).expand(-1, -1, self.in_features)
        pre_ = pre.view(batch_size, 1, self.in_features).expand(-1, self.out_features, -1)

        if self.hebb_mode is None or self.hebb_mode in ['pointwise', 'affine']:
            prepost = post_ * pre_  # (batch_size, out_features, in_features)
        elif self.hebb_mode == 'mlp':
            post_ = torch.reshape(post_, (-1, 1))  # (batch_size*out_features*in_features, 1)
            pre_ = torch.reshape(pre_, (-1, 1))
            prepost = torch.cat((pre_, post_), dim=1)
            prepost = self.hebb_fn(prepost)
            prepost = prepost.view(batch_size, self.out_features, self.in_features)
        else:
            raise ValueError('Unknown hebb mode', str(self.hebb_mode))

        if third_factor is None:
            hebb_update_rate = self.hebb_update_rate
        else:
            if len(third_factor.shape) == 2:
                # shape (batch_size, 1) or (batch_size, out_features)
                # Third factor is either uniform, or post-synaptic
                assert third_factor.shape[0] == batch_size
                if third_factor.shape[1] == 1:
                    _third_factor = third_factor.view((batch_size, 1, 1))
                else:
                    if thirdfactor_syn == 'post':
                        assert third_factor.shape[1] == self.out_features
                        _third_factor = third_factor.view(
                            (batch_size, self.out_features, 1))
                    elif thirdfactor_syn == 'pre':
                        assert third_factor.shape[1] == self.in_features
                        _third_factor = third_factor.view(
                            (batch_size, 1, self.in_features))
                    else:
                        raise ValueError('Unknown thirdfactor_syn',
                                         thirdfactor_syn)

                # make (batch_size, 1, 1) or (batch_size, out_features, 1)
                hebb_update_rate = self.hebb_update_rate * _third_factor
            else:
                # third factor should be (batch_size, out_features, in_features)
                hebb_update_rate = self.hebb_update_rate * third_factor

        # Below will hurt performance very much, not sure why
        # hebb_update_rate = torch.sigmoid(hebb_update_rate)

        # hebb_update_rate = torch.clamp(hebb_update_rate, min=0.0, max=1.0)

        if self.stability:
            hebb_update_rate = hebb_update_rate * (1 - self.hebb_stability)
            deltahebb_ = torch.clamp(hebb_update_rate,min=0.0, max=1.0) * prepost
        else:
            deltahebb_ = hebb_update_rate * prepost

        if self.decay == 'active':
            decay_rate = 1 - torch.clamp(hebb_update_rate, min=0.0, max=1.0)
        elif self.decay is None:
            decay_rate = self.decay_rate
        elif self.decay.startswith('passive') and third_factor is not None:
            #if third_factor is 0 then don't decay, if 1 then decay as normal
            decay_rate = 1 - (1-self.decay_rate)*third_factor
        else:
            raise ValueError('Invalid decay: {}'.format(self.decay))

        self.hebb = torch.clamp(decay_rate*self.hebb + deltahebb_,
                                min=-self.hebb_clamp, max=self.hebb_clamp)

        if self.stability:
            # Update stability tensor
            # TODO: Set proper stability update threshold
            self.hebb_stability = (1.0 * self.hebb_stability +
                                   (hebb_update_rate >= self.stability_threshold) * 1.0)
            self.hebb_stability = torch.clamp(
                self.hebb_stability, min=0., max=1.)

        if self.normalize:
            norm = self.hebb.norm(dim=self.normalize_dim).unsqueeze(self.normalize_dim)
            self.hebb = self.normalize_scale * self.hebb / (norm+EPSILON)

        if self.analyzing:
            self.writer['third_factor'].append(third_factor)
            self.writer['hebb'].append(self.hebb)
            self.writer['hebb_update_rate'].append(hebb_update_rate)
            self.writer['prepost'].append(prepost)
            self.writer['deltahebb'].append(deltahebb_)
            if self.stability:
                self.writer['hebb_stability'].append(self.hebb_stability)

    def forward(self, input):
        """

        Args:
             input: tensor (batch_size, input_features)

        Return:
            output: tensor (batch_size, output_features)
        """
        # Here, the *rows* of w and hebb are the inputs weights to a single neuron
        # hidden = x, hactiv = y
        batch_size = input.shape[0]

        # effective weight (out_features, in_features)
        w = torch.mul(self.hebb_strength, self.hebb)
        if self.weight is not None:
            w = self.weight + w
        output_ = torch.matmul(w, input.view(batch_size, self.in_features, 1))

        return output_.squeeze(dim=2)
