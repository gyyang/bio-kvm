# Define task

from _collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

import sys
import os

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools


class RecallDataset(object):

    def __init__(
            self,
            stim_dim=10,
            T_min=15,
            T_max=25,
            T_distribution='uniform',
            p_recall=0.1,
            chance=0.7,
            balanced=True,
            temporal_corr=0.,
            temporal_corr_mode='template',
            spatial_corr=0.,
            n_repeat=1,
            sigma=0.,
            recall_order='sequential',
            recall_interleave_delay=None,
            heteroassociative=False,
            heteroassociative_stim_dim=None,
            **kwargs,
    ):
        """
        Args:
            stim_dim: int, stimulus dimension
            T_min: Minimum stored sequence length
            T_max: Maximum stored sequence length
            p_recall: proportion of patterns stored for recall
            chance: chance level performance
            balanced: bool, if True, input is on average centered around 0
            n_repeat: int, if larger than 1, then inputs are repeated in time
            temporal_corr: float, between 0 and 1, correlation among stimuli
            temporal_corr_mode: str, 'template' (generates a correlated dataset
                by perturbung a template vector) or 'drift' (generates the
                (t+1)th input by perturbing the t-th input)
            spatial_corr: float, spatial correlation of stimuli, between 0 and 1
            sigma: float, strength of input noise
            recall_order: str, order of store, recall.
                'sequential' or 'interleave'
            recall_interleave_delay: int, interval after initial presentation
                after which to prompt the recall (only for recall_order=='interleave')
        """
        super(RecallDataset, self).__init__()

        self.stim_dim = stim_dim
        self.store_signal_dim = 1  # can only be 1 now
        self.input_dim = self.stim_dim + self.store_signal_dim
        self.output_dim = heteroassociative_stim_dim if heteroassociative_stim_dim else stim_dim
        self.T_min = int(np.ceil(T_min/n_repeat))
        if T_max is None:
            self.T_max = T_min
        else:
            self.T_max = int(np.ceil(T_max/n_repeat))
        assert self.T_max >= self.T_min, 'T_max must be larger than T_min'
        if self.T_min*n_repeat != T_min or self.T_max*n_repeat != T_max:
            print('Warning: T_min and/or T_max not divisible by n_repeat. Rounding '
             'T_min={}->{}, T_max={}->{}'.format(T_min, self.T_min*n_repeat,
                                                   T_max, self.T_max*n_repeat))
        self.T_distribution = T_distribution
        if T_distribution == 'uniform':
            self.generate_T = lambda: np.random.randint(self.T_min, self.T_max + 1)
        else:
            raise ValueError('Not supported T distribution type', str(T_distribution))
        self.p_recall = p_recall
        self.balanced = balanced
        self.chance = chance
        if self.balanced:
            self.p_unknown = (1 - chance) * 2.
        else:
            # p_flip: amount of sensory noise during recall
            self.p_flip = 1 - chance
        self.n_repeat = n_repeat
        self.spatial_corr = spatial_corr
        self.temporal_corr = temporal_corr
        self.temporal_corr_mode = temporal_corr_mode
        self.sigma = sigma
        self.recall_order = recall_order
        self.recall_interleave_delay = recall_interleave_delay
        self.heteroassociative = heteroassociative
        self.heteroassociative_stim_dim = heteroassociative_stim_dim

    def __str__(self):
        print('Recall dataset:')
        nicename_dict = OrderedDict(
            [('stim_dim', 'Stimulus dimension'),
             ('store_signal_dim', 'Storage signal dimension'),
             ('input_dim', 'Input dimension'),
             ('output_dim', 'Output dimension'),
             ('T_min', 'Minimum stored sequence length'),
             ('T_max', 'Maximum stored sequence length'),
             ('T_distribution', 'Sequence length distribution'),
             ('p_recall', 'Proportion of recall'),
             ('chance', 'Chancel level accuracy'),
             ('n_repeat', 'Number of input repeats'),
             ('spatial_corr', 'Spatial correlation factor'),
             ('sigma', 'Input noise level'),
             ]
        )
        if self.balanced:
            nicename_dict['p_unknown'] = 'Proportion of unknown elements at recall'
        else:
            nicename_dict['p_flip'] = 'Proportion of flipping at recall'

        string = ''
        for key, name in nicename_dict.items():
            string += name + ' : ' + str(getattr(self, key)) + '\n'
        return string

    def make_time_indices(self):
        """Generate the time indices for the data.
        :returns
            store_ind: array of indices for stimuli time
            recall_ind: array of indices for recall time
            store_recall_map: array with two rows [s1, s2, s3,
                                                   r1, r2, r3]
                at time r1, network should recall stimulus stored at time s1
        """
        T_store = self.generate_T()  # store phase length
        #T_recall = int(self.p_recall * T_store)  # recall phase length
        T_recall = max(1, int(self.p_recall * T_store)) ##needed for the case T = 1 and p_recall < 1
        T = T_store + T_recall
        p_store = T_store * 1.0 / T

        if self.recall_order == 'sequential':
            store_ind = np.arange(T_store)
            recall_ind = np.arange(T_store, T_store + T_recall)

            store_recall_map = np.array([
                np.random.choice(np.arange(T_store), T_recall, replace=False),
                recall_ind
            ])
        elif self.recall_order == 'interleave':
            store_ind = list()
            recall_ind = list()
            store_recall_map_0 = list()
            store_recall_map_1 = list()

            if self.recall_interleave_delay:  # recall stim from exactly R timesteps ago
                for i in range(T):
                    if (i < self.recall_interleave_delay  # first R always new
                            or np.random.rand() > self.p_recall  # otherwise p_recall chance of repeat
                            or i - self.recall_interleave_delay in recall_ind):  # but no double repeats
                        store_ind.append(i)
                    else:
                        recall_ind.append(i)
                        assert i - self.recall_interleave_delay in store_ind  # sanity check
                        store_recall_map_0.append(i - self.recall_interleave_delay)
                        store_recall_map_1.append(i)

            else:  # recall can be any previously seen stimulus
                for i in range(T):
                    if (i < T - 1) and (np.random.rand() < p_store or i == 0):
                        store_ind.append(i)
                    else:
                        recall_ind.append(i)
                        # Select a stored stimulus to recall
                        store_recall_map_0.append(np.random.choice(store_ind))
                        store_recall_map_1.append(i)

            store_ind = np.array(store_ind)
            recall_ind = np.array(recall_ind)
            store_recall_map = np.array([store_recall_map_0,
                                         store_recall_map_1])

        else:
            raise ValueError('Unknown recall order', self.recall_order)

        return store_ind, recall_ind, store_recall_map

    def generate(self):
        """Generate one batch of data.
        Return:
            data: dictionary with entries:
                input: network input, np array (time, stim_dim)
                modu_input: modulatory input, np array (time, store_signal_dim)
                target: target output, np array (time, stim_dim)
                mask: mask, np array (time,). Used for computing loss
        """
        stim_dim = self.stim_dim

        store_ind, recall_ind, store_recall_map = self.make_time_indices()

        T_store = len(store_ind)
        T_recall = len(recall_ind)
        T = T_store + T_recall

        X_stim = np.zeros((T, stim_dim))

        if self.heteroassociative:
            Y = np.zeros((T, self.heteroassociative_stim_dim))
        else:
            Y = np.zeros((T, stim_dim))

        M = np.zeros(T)

        if self.heteroassociative:
            X_stim_targ = np.zeros((T, self.heteroassociative_stim_dim))

        # Storage phase
        if self.balanced:
            X_vals = (-1, 1)
        else:
            X_vals = (0, 1)

        # Optional spatial xor temporal correlation
        if self.spatial_corr > 0:
            if self.temporal_corr != 0:
                raise NotImplementedError('Simultaneous spatial and temporal '
                                          'correlation not implemented')
            if self.heteroassociative:
                raise NotImplementedError('Heteroassociative recall '
                                          'not implemented with correlations')
            X_stim[store_ind, 0] = np.random.choice(X_vals, size=len(store_ind))
            for i in range(1, stim_dim):
                X_stim[store_ind, i] = X_stim[store_ind, i - 1]
                redraw_idx = store_ind[np.random.rand(len(store_ind)) >= self.spatial_corr]
                X_stim[redraw_idx, i] = np.random.choice(X_vals, size=len(redraw_idx))
        elif self.temporal_corr > 0:
            if self.spatial_corr != 0:
                raise NotImplementedError('Simultaneous spatial and temporal '
                                          'correlation not implemented')
            if self.heteroassociative:
                raise NotImplementedError('Heteroassociative recall '
                                          'not implemented with correlations')
            X_stim_0 = np.random.choice(X_vals, size=stim_dim)
            X_stim[store_ind[0]] = X_stim_0
            for i in range(1, len(store_ind)):
                if self.temporal_corr_mode == 'template':
                    X_stim[store_ind[i]] = X_stim_0
                elif self.temporal_corr_mode == 'drift':
                    X_stim[store_ind[i]] = X_stim[store_ind[i-1]]
                else:
                    raise ValueError('Invalid temporal_corr_mode: {}'.format(
                        self.temporal_corr_mode))
                redraw_idx = np.random.rand(stim_dim) >= self.temporal_corr
                X_stim[store_ind[i], redraw_idx] = np.random.choice(X_vals,
                                                        size=redraw_idx.sum())
        else:
            X_stim[store_ind] = np.random.choice(X_vals, size=(len(store_ind),
                                                               stim_dim))
            if self.heteroassociative:
                X_stim_targ[store_ind] = np.random.choice(
                    X_vals, size=(len(store_ind), self.heteroassociative_stim_dim)
                    )

        # # If to be recalled, set store signal to True
        X_store_signal = np.zeros((T, self.store_signal_dim))
        X_store_signal[store_recall_map[0], :] = 1.

        #Always store, except during a query
        # X_store_signal = np.ones((T, self.store_signal_dim))
        # X_store_signal[store_recall_map[1], :] = 0.





        # Recall phase
        X_stim_recall = X_stim[store_recall_map[0]]  # pre-perturbation
        if self.heteroassociative:
            Y[store_recall_map[1], :] = X_stim_targ[store_recall_map[0]]
        else:
            Y[store_recall_map[1], :] = X_stim_recall  # target is original
        M[store_recall_map[1]] = 1.

        # Perturb X_stim_recall
        # Flip probability
        if self.balanced:
            known_matrix = (np.random.rand(T_recall, stim_dim) >
                            self.p_unknown) * 1.0
            X_stim_recall = X_stim_recall * known_matrix
        else:
            flip_matrix = np.random.rand(T_recall, stim_dim) < self.p_flip
            X_stim_recall = (X_stim_recall * (1 - flip_matrix) +
                             (1 - X_stim_recall) * flip_matrix)
        X_stim[store_recall_map[1], :stim_dim] = X_stim_recall

        data = {
            'input': X_stim,
            'modu_input': X_store_signal,
            'target': Y,
            'mask': M,
            'store_ind': store_ind,
            'recall_ind': recall_ind,
            'store_recall_map': store_recall_map
        }

        if self.heteroassociative:
            data['input_heteroassociative'] = X_stim_targ

        # Optional repeat inputs (another version of temporal correlation)
        if self.n_repeat > 1:
            for key, val in data.items():
                data[key] = np.repeat(val, self.n_repeat, axis=0)
            data['store_ind'] *= self.n_repeat
            data['recall_ind'] *= self.n_repeat
            data['store_recall_map'] *= self.n_repeat

        # Add optional noise
        if self.sigma > 0:
            for key in ['input', 'modu_input']:
                data[key] += np.random.randn(*data[key].shape) * self.sigma

        return data


    def visualize(self, figpath=None, figname=None, title_suffix='',
                  figsize_scalings=None, vmin=None, vmax=None):
        """Helper function to visually explain the dataset structure."""

        data = self.generate()

        X = np.concatenate((data['input'], data['modu_input']), axis=1)
        Y = data['target']
        M = data['mask']
        matrices_to_plot = [X, Y]
        plottype = ['input', 'target', 'input_Y']
        if 'input_heteroassociative' in data.keys():
            input_Y = data['input_heteroassociative']
            matrices_to_plot.append(input_Y)

        for i, val in enumerate(matrices_to_plot):
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
            elif i == 1:
                title = 'Target Y' + title_suffix
                ylabel = 'Target dimension'
            else:
                title = 'Input Y' + title_suffix
                ylabel = 'Stimulus dimension'

            plt.title(title, fontsize=7)
            plt.xlabel('Time step', labelpad=15)
            plt.ylabel(ylabel, labelpad=15)
            plt.xticks([])
            plt.yticks([])

            colors = np.array([[55, 126, 184], [228, 26, 28], [178, 178, 178]]) / 255
            labels = M  # From dataset
            texts = ['Store', 'Test']
            # TODO: Not working for recall_order='interleave'
            tools.add_colorannot(fig, rect_bottom, labels, colors, texts)

            if i == 0:
                # Note: Reverse y axis
                # Innate, flexible
                colors = np.array([[245, 110, 128], [149, 0, 149]]) / 255
                labels = np.array([1] * self.store_signal_dim + [0] * self.stim_dim)
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
                figname = figname + '_' + plottype[i]
                tools.save_fig(figpath, figname, pdf=True)


class EcstasyRecall(RecallDataset):
    def __init__(self, p_ec=0.1, ec_strength=10, num_ec=None, **kwargs):
        """
        :param p_ec: probablity of ecstatic inputs
        :param ec_strength: strength of the gating variable compared to normal
        """
        super().__init__(**kwargs)
        self.p_ec = p_ec
        self.ec_strenth = ec_strength
        self.num_ec = num_ec

    def generate(self):
        data = super().generate()
        # indices for storing inputs
        store_recall_map = data['store_recall_map']
        store_ind = store_recall_map[0]
        # indices for ecstatic inputs
        if self.num_ec is None:
            num_ec = int(len(store_ind) * self.p_ec)
        else:
            num_ec = self.num_ec
        size_ec = max((1, num_ec))

        _ec_ind = np.random.choice(len(store_ind), size=size_ec, replace=False)
        _nonec_ind = [ind for ind in range(len(store_ind)) if ind not in _ec_ind]
        self.ec_store_recall_map = [store_recall_map[0][_ec_ind],
                                    store_recall_map[1][_ec_ind]]
        self.nonec_store_recall_map = [store_recall_map[0][_nonec_ind],
                                    store_recall_map[1][_nonec_ind]]
        self.ec_ind = self.ec_store_recall_map[0]

        # Modify store signal for ecstatic inputs
        data['modu_input'][self.ec_ind] *= self.ec_strenth
        data['ec_store_recall_map'] = np.array(self.ec_store_recall_map)
        data['nonec_store_recall_map'] = np.array(self.nonec_store_recall_map)
        return data


def plot_spatial_corr(ax=None, X=None, spatial_corr=0.5, T=1000, **dataset_kwargs):
    """
    Plot normalized spatial autocorrelation at all lags
    """
    if X is None:
        dataset = RecallDataset(T_min=T, T_max=T,
                                spatial_corr=spatial_corr,
                                store_signal_dim=1, p_recall=0,
                                **dataset_kwargs)
        X, _, _ = dataset.generate()
        X = X[:, :-1]

    corr = np.array([[np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, len(x) - 1)] for x in X])
    avg_corr = np.nanmean(corr, axis=0)

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(3, 2)
    ax.plot(avg_corr, label=spatial_corr)
    ax.legend()
    ax.set_xlabel('Spatial lag')
    ax.set_ylabel('Autocorrelation')
    ax.get_figure().tight_layout()
    return ax


if __name__ == '__main__':
    # dataset = RecallDataset(T_min=10, T_max=10, stim_dim=30, p_recall=1,
    #                         n_repeat=1, sigma=0.,
    #                         # temporal_corr=0.7, temporal_corr_mode='drift',
    #                         # recall_order='interleave', recall_interleave_delay=3
    #                         )
    # dataset.visualize()


    # dataset = EcstasyRecall(T_min=15, T_max=15, stim_dim=30, p_recall=0.5,
    #                         recall_order='interleave', recall_interleave_delay=3,
    #                         ec_strength=3, p_ec=0.3)
    # dataset.visualize()
    import configs
    config = configs.get_config('ref_seq', stim_dim=40, hidden_size=40)
    config['dataset']['recall_order'] = 'interleave'
    config['dataset']['recall_interleave_delay'] = 3
    config['dataset']['T_min'] = config['dataset']['T_max'] = 60
    config['dataset']['p_recall'] = 0.5
    dataset = RecallDataset(**config['dataset'])
    dataset.visualize()
