# Training
import shutil
import os
import sys
import time
import pickle
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
# import analysis.analysis as analysis
import analysis

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'


def _plot(X, Y, value, C):
    """Specialized plotting function."""
    figsize = (2, 2)
    ax_box = (0.2, 0.3, 0.6, 0.6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_box)
    
    # Create a set of line segments so that we can color them individually
    points = np.concatenate((X[:, :, np.newaxis], Y[:, :, np.newaxis]), -1)
    # Add dimension for stacking X and Y together easily to get the segments. 
    points = points[:, :, np.newaxis, :]
    # The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    segments = np.concatenate([points[:-1], points[1:]], axis=2)
    # Put together dimension for conditions and for time
    segments = np.reshape(segments, (-1, 2, 2))
    
    # Create a continuous norm to map from data points to colors
    lc = LineCollection(segments)
    # Set the values used for colormapping
    lc.set_array(value)
    lc.set_color(C)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    cbar_ax = fig.add_axes([0.82, 0.3, 0.02, 0.6])
    cb = fig.colorbar(line, cax=cbar_ax)
    cb.set_label('Training accuracy', labelpad=-6)
    lim = value.min(), value.max()
    cb.set_ticks(lim)
    cb.set_ticklabels(['{:0.2f}'.format(l) for l in lim])
    
    # print(C.shape)
     #print(segments.shape)
    
# =============================================================================
#     i_max = np.argmax(C)
#     point_max = segments[i_max, 1, :]
#     ax.plot(*point_max, marker='^', markersize=3,
#             color=np.array([228,26,28])/255.)
# =============================================================================
    
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Pre-synaptic\ndependence', labelpad=-5)
    ax.set_ylabel('Post-synaptic\ndependence', labelpad=-5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([0, 1.0])
    ax.set_yticks([0, 1.0])
    return fig, ax, lc
    

def plot_prepost_hebbprogress(save_path, select=None, exclude=None, movie=False):
    # First check if model has
    log = tools.load_results(save_path, get_last=False,
                             select=select, exclude=exclude)
    
    word = '.pre_fn.weight'
    hebb_layers = [key[:-len(word)] for key in log.keys() if word in key]
    if not hebb_layers:
        print('No affine PlasticLinear layer found')
        return
        
    acc = log['acc_train'][:, :-1]
    n_network, n_time = acc.shape
    acc = np.reshape(acc.T, (-1,))
    
    cmap = mpl.cm.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=acc.min(), vmax=acc.max())
    C = cmap(norm(acc))
        
    for hebb_layer in hebb_layers:
        hebbness = dict()
        for side in ['pre', 'post']:
            tmp = hebb_layer + '.' + side + '_fn.'
            a, b = log[tmp+'bias'], log[tmp+'weight']
            hebbness[side] = abs(b) / (abs(a) + abs(b))
    
        X = hebbness['pre'].T
        Y = hebbness['post'].T
        
        fig, ax, lc = _plot(X, Y, acc, C)
        ax.scatter(X[0, :], Y[0, :], s=5, c=C[:n_network])
        
        title = tools.nicename(hebb_layer)
        title = title.split()[-1] + ' Plasticity'
        
        ax.set_title(title, fontsize=7)
        figname = 'prepost_hebbprogress' + hebb_layer
        
        if not movie:
            tools.save_fig(save_path, figname)
            continue
        
        
        def animate(i):
            c = C.copy()    
            if i >= 2:
                # set c after time i to have alpha value 0
                c[(i-2)*n_network:, 3] = 0
                lc.set_color(c)
            else:
                c[:, 3] = 0
                lc.set_color(c)
            return ax
    
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate,
                                       frames=n_time+2, interval=20)
        writer = animation.writers['ffmpeg'](fps=30)
        figname = 'prepost_hebbprogress' + hebb_layer
        anim.save(os.path.join(save_path.replace('files', 'figures'),
                               figname+'_movie.mp4'), writer=writer, dpi=600)


def evaluate_network_size(save_path, sizes=None):
    """Evaluate network at different network sizes."""
    if sizes is None:
        sizes = [10, 20, 40, 100, 200, 400, 800, 1000]
    for size in sizes:
        update_config = {'plasticnet': {'hidden_size': size}}
        analysis.evaluate_run(modeldir=save_path, n_batch=100,
                              update_config=update_config,
                              fname='hidden{:d}'.format(size), load_hebb=False)


def plot_network_size_acc(save_path, sizes=None):
    if sizes is None:
        sizes = [10, 20, 40, 100, 200, 400, 800, 1000]

    # TODO: it's awkward needing to do all this
    res = defaultdict(list)
    for size in sizes:
        fname = 'hidden{:d}.pkl'.format(size)
        with open(os.path.join(save_path, fname), 'rb') as f:
            results = pickle.load(f)
        for key, val in results.items():
            if key == 'config':
                val = tools.flatten_nested_dict(val)
                for k, v in val.items():
                    res[k].append(v)
            elif key == 'acc':
                # Get accuracy at recall ind
                n_batch = len(results['acc'])
                acc = list()
                for i_batch in range(n_batch):
                    recall_ind = results['store_recall_map'][i_batch][1]
                    acc_tmp = results['acc'][i_batch][recall_ind].mean()
                    acc.append(acc_tmp)
                res[key].append(acc)
            else:
                res[key].append(val)

    analysis.plot_results(save_path, 'plasticnet.hidden_size', 'acc', res=res,
                          logx=True, figsize=(2.5, 1.2))


if __name__ == '__main__':
    # name = 'p_random'
    name = 'affine_init'
    select = {'hebb_config.mode': 'affine'}
    exclude = None
    save_path = os.path.join('../files/', name)
    plot_prepost_hebbprogress(save_path, select=select, movie=True)

        
        
