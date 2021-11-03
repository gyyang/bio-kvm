from _collections import OrderedDict
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tools

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

class SequenceRecallDataset(object):
    """
    Like the Copy task, the network is first fed a sequence of patterns. After
    the end-of-sequence marker, the network will be given a prompt. This will
    be a one the patterns in the sequence. To correctly
    do the task, the network should output the rest of the sequence that follows
    the prompt pattern.
    """

    def __init__(
            self, pattern_dim=40, n_patterns_min=2, n_patterns_max=10,
            prompt_start_min=0, prompt_start_max=0.75, version=0,
            modu_input=True, eos=False, balanced=True, truncate=False,
            constant_mask=False, **kwargs
    ):
        """
            pattern_dim: int, dimension of each pattern
            n_patterns_min: int, min number of patterns to copy
            n_patterns_max: int, max number of patterns to copy
            prompt_start_min: float, 0-1; min start index of prompt
            prompt_start_min: float, 0-1; max start index of prompt
            modu_input: bool, whether to add a channel for modulatory input
        """

        super(SequenceRecallDataset, self).__init__()
        self.pattern_dim = pattern_dim
        self.n_patterns_min = n_patterns_min
        self.n_patterns_max = n_patterns_max
        self.prompt_start_min = prompt_start_min
        self.prompt_start_max = prompt_start_max
        self.input_dim = pattern_dim
        self.output_dim = pattern_dim
        #self.input_dim += 1 # With channel
        self.modu_input = modu_input
        self.version = version
        self.chance = 0.5
        self.balanced = balanced

    def __str__(self):
        print('Sequence recall dataset:')
        param_name_dict = OrderedDict(
            [('pattern_dim', 'Pattern dimension'),
             ('n_patterns_min', 'Min number of patterns'),
             ('n_patterns_max', 'Max number of patterns'),
             ('prompt_start_min', 'Min start index of prompt'),
             ('prompt_start_max', 'Max start index of prompt')
            ]
        )
        string = ''
        for key, name in param_name_dict.items():
            string += name +' : ' + str(getattr(self, key)) + '\n'
        return string

    def generate(self, n_patterns=None):
        """
        Generates a batch of data

        :returns
            X: network input, np array (copy len+paste len, input_dim)
            Y: target output, np array (copy len+paste len, output_dim)
            M: mask, np array, specifies which timesteps loss should be calculated
        """

        if n_patterns is None:
            n_patterns = np.random.randint(
                self.n_patterns_min, self.n_patterns_max+1
                )
        prompt_start = np.random.randint(0, n_patterns-1)

        copy_len = n_patterns
        recall_len = (n_patterns-prompt_start) # With delay
        

        if self.balanced:
            X = np.ones((copy_len + recall_len, self.input_dim))*-1
            Y = np.ones((copy_len + recall_len, self.output_dim))*-1
        else:
            X = np.zeros((copy_len + recall_len, self.input_dim))
            Y = np.zeros((copy_len + recall_len, self.output_dim))

        M = np.zeros(X.shape[0])
        modu_input = np.zeros((copy_len + recall_len, 1))
        modu_input[:copy_len+1] = 1
        if self.balanced:
            patterns = np.random.choice((-1, 1), size=(n_patterns, self.pattern_dim))
        else:
            patterns = np.random.choice((0, 1), size=(n_patterns, self.pattern_dim))

        X[:copy_len, :] = patterns
        X[copy_len, :] = patterns[prompt_start,:]
        Y[:copy_len, :] = patterns
        Y[copy_len:, :] = patterns[prompt_start:,:]
        M[copy_len:] = 1

        X_paste_window = X[copy_len+1:]
        X_paste_window[X_paste_window == -1] = 0

        data = {'input': X, 'target': Y, 'mask': M}
        if self.modu_input:
            data['modu_input'] = modu_input
        return data


    def visualize(self, figpath=None, figname=None, title_suffix='',
                  figsize_scalings=None, vmin=None, vmax=None):
        """Helper function to visually explain the dataset structure."""

        data = self.generate()

        X = np.concatenate((data['input'], data['modu_input']), axis=1)
        Y = data['target']
        Y[data['modu_input'].squeeze() == 1,:] = 0
        M = data['mask']
        M[np.sum(data['modu_input'].squeeze() == 1)-1] = 0.5

        for i, val in enumerate([X, Y]):
            if figsize_scalings is None:
                figsize = (3., 2.)
            else:
                figsize = (3*figsize_scalings[i][0], 2*figsize_scalings[i][1])
            rect = [0.15, 0.15, 0.65, 0.65]
            rect_cb = [0.82, 0.15, 0.02, 0.65]
            rect_bottom = [0.15, 0.12, 0.65, 0.02]
            rect_left = [0.12, 0.15, 0.02, 0.65]

            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes(rect)
            if vmin is None:
                vmin = -np.abs(val).max()
            if vmax is None:
                vmax = np.abs(val).max()

            im = plt.imshow(val.T, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)

            if i == 0:
                title = 'Input X' + title_suffix
                ylabel = 'Stimulus dimension'
            else:
                title = 'Target Y' + title_suffix
                ylabel = 'Target dimension'
            plt.title(title, fontsize=7)
            plt.xlabel('Time step', labelpad=15)
            plt.ylabel(ylabel, labelpad=15)
            plt.xticks([])
            plt.yticks([])

            colors = np.array(
                [[55, 126, 184], [178, 178, 178], [228, 26, 28]]
                ) / 255
            labels = M  # From dataset
            texts = ['Store', 'Prompt', 'Test']
            tools.add_colorannot(fig, rect_bottom, labels, colors, texts)

            if i == 0:
                # Note: Reverse y axis
                # Innate, flexible
                colors = np.array([[245, 110, 128], [149, 0, 149]]) / 255
                labels = np.array([1] * 1 + [0] * self.pattern_dim)
                texts = ['Stimulus', 'Gate']
                tools.add_colorannot(fig, rect_left, labels, colors, texts,
                                     orient='vertical')

            ax = fig.add_axes(rect_cb)
            cb = plt.colorbar(im, cax=ax)
            cb.outline.set_linewidth(0.5)
            cb.set_ticks([vmin, vmax])
            cb.set_label('Activity', labelpad=-10)
            plt.tick_params(axis='both', which='major')

            # plt.tight_layout()

            if figpath is not None:
                plottype = 'input' if i == 0 else 'target'
                figname = figname + '_' + plottype
                tools.save_fig(figpath, figname, pdf=True)

