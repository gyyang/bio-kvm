from _collections import OrderedDict
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tools

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

class CopyPasteDataset(object):
    """
    Implemented as in the Repeat/Copy task in the Neural Turing Machine paper.
    If n_paste_min and n_paste_max are both set to 1, mimics the Copy task of
    the NTM paper. Inputs currently do not have a modulatory input channel by
    default but do have a channel for the end-of-sequence marker (which
    indicates the number of repeats).
    """

    def __init__(
            self, pattern_dim=8,
            n_patterns_min=1, n_patterns_max=5, n_paste_min=1, n_paste_max=5,
            normalize_to_min=None, normalize_to_max=None,
            modu_input=False, balanced=True, **kwargs
    ):
        """
            pattern_dim: int, dimension of each pattern
            n_patterns_min: int, min number of patterns to copy
            n_patterns_max: int, max number of patterns to copy
            n_paste_min: int, min number of pasting (repeating) in the output
            n_paste_max: int, max number of pasting (repeating) in the output
            modu_input: bool, whether to add a channel for modulatory input
        """

        super(CopyPasteDataset, self).__init__()
        self.pattern_dim = pattern_dim
        self.n_patterns_min = n_patterns_min
        self.n_patterns_max = n_patterns_max
        self.n_paste_min = n_paste_min
        self.n_paste_max = n_paste_max
        self.input_dim = pattern_dim + 1
        self.output_dim = pattern_dim + 1
        self.modu_input = modu_input
        self.balanced = balanced

        if normalize_to_min is None:
            normalize_to_min = n_paste_min
        if normalize_to_max is None:
            normalize_to_max = n_paste_max

        self.paste_mean = (normalize_to_min + normalize_to_max)/2
        paste_var = ((normalize_to_max - normalize_to_min + 1) - 1)/12
        self.paste_std = np.sqrt(paste_var)
        self.chance = 0.5

    def __str__(self):
        print('Copy/Paste dataset:')
        param_name_dict = OrderedDict(
            [('pattern_dim', 'Pattern dimension'),
             ('n_patterns_min', 'Min number of patterns'),
             ('n_patterns_max', 'Max number of patterns'),
             ('n_paste_min', 'Min number of pastes'),
             ('n_paste_max', 'Max number of pastes')
            ]
        )
        string = ''
        for key, name in param_name_dict.items():
            string += name +' : ' + str(getattr(self, key)) + '\n'
        return string

    def generate(self, n_patterns=None, n_paste=None):
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
        if self.n_paste_min == self.n_paste_max:
            n_paste = self.n_paste_min
        else:
            if n_paste is None:
                n_paste = np.random.randint(
                    self.n_paste_min, self.n_paste_max+1
                    )

        copy_len = n_patterns + 1
        paste_len = n_patterns*n_paste + 1
        X = np.zeros((copy_len + paste_len, self.input_dim))
        Y = np.zeros((copy_len + paste_len, self.output_dim))
        M = np.zeros(X.shape[0])
        modu_input = np.zeros((copy_len + paste_len, 1))
        modu_input[:copy_len] = 1
        if self.balanced:
            patterns = np.random.choice((-1, 1), size=(n_patterns, self.pattern_dim))
        else:
            patterns = np.random.choice((0, 1), size=(n_patterns, self.pattern_dim))
        X[:copy_len-1, :-1] = patterns
        X[copy_len-1, -1] = self._normalize_paste(n_paste)
        Y[copy_len:-1, :-1] = np.tile(patterns, (n_paste, 1))
        Y[-1, -1] = 1
        M[copy_len:] = 1

        if self.balanced:
            X[X == 0] = -1
            X_paste_window = X[copy_len:]
            X_paste_window[X_paste_window == -1] = 0
            Y_paste_window = Y[copy_len:]
            Y_paste_window[Y_paste_window == 0] = -1
        data = {'input': X, 'target': Y, 'mask': M}
        if self.modu_input:
            data['modu_input'] = modu_input
        return data

    def _normalize_paste(self, n_paste):
        """
        Normalizes repeat number to have mean 0 and variance 1 within the
        specified min/max interval.
        """

        if self.n_paste_min == self.n_paste_max:
            return 1
        return (n_paste - self.paste_mean)/self.paste_std

    def visualize(self, figpath=None, figname=None):
        """Helper function to visually explain the dataset structure."""

        data = self.generate()

        X = data['input']
        Y = data['target']
        M = data['mask']

        for i, data in enumerate([X, Y]):
            figsize = (3., 2.)
            rect = [0.15, 0.15, 0.65, 0.65]
            rect_cb = [0.82, 0.15, 0.02, 0.65]
            rect_bottom = [0.15, 0.12, 0.65, 0.02]
            rect_left = [0.12, 0.15, 0.02, 0.65]

            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes(rect)
            im = plt.imshow(data.T, aspect='auto', cmap='RdBu_r')

            if i == 0:
                title = 'Input X'
                ylabel = 'Stimulus dimension'
            else:
                title = 'Target Y'
                ylabel = 'Target dimension'
            plt.title(title, fontsize=7)
            plt.xlabel('Time step', labelpad=15)
            plt.ylabel(ylabel, labelpad=15)
            plt.xticks([])
            plt.yticks([])

            colors = np.array([[55, 126, 184], [228, 26, 28], [178, 178, 178]]) / 255
            labels = M  # From dataset
            texts = ['Store', 'Test']
            tools.add_colorannot(fig, rect_bottom, labels, colors, texts)

            if i == 0:
                # Note: Reverse y axis
                # Innate, flexible
                colors = np.array([[245, 110, 128], [149, 0, 149]]) / 255
                labels = np.array([1] + [0]*self.pattern_dim)
                texts = ['Stimulus', 'EOS']
                tools.add_colorannot(fig, rect_left, labels, colors, texts,
                                     orient='vertical')

            ax = fig.add_axes(rect_cb)
            cb = plt.colorbar(im, cax=ax)
            cb.outline.set_linewidth(0.5)
            vmin = -np.abs(data).max()
            vmax = np.abs(data).max()
            cb.set_ticks([vmin, vmax])
            cb.set_label('Activity', labelpad=-10)
            plt.tick_params(axis='both', which='major')
            # plt.tight_layout()

            if figpath is not None:
                plottype = 'input' if i == 0 else 'target'
                figname = figname + '_' + plottype
                tools.save_fig(figpath, figname, pdf=True)
