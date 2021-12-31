"""
Calculate the information in the network
"""

from multiprocessing import cpu_count
from joblib import Parallel, delayed

import warnings
import numpy as np
import numba
import numpy as np
#import pytorch_network
from collections import OrderedDict
import torch
from torch import nn
import tqdm

#from utils import get_aligned_representations
#from plot_information import plot_information_plane
from sklearn.model_selection import train_test_split
import importlib

#import plot_information
import matplotlib.pyplot as plt

from activation_trackers import ActivationTracker

NUM_CORES = cpu_count()
warnings.filterwarnings("ignore")


@numba.jit
def entropy(probs):
    return -np.sum(probs * np.ma.log2(probs))


@numba.jit
def joint_entropy(unique_inverse_x, unique_inverse_y, bins_x, bins_y):

    joint_distribution = np.zeros((bins_x, bins_y))
    np.add.at(joint_distribution, (unique_inverse_x, unique_inverse_y), 1)
    joint_distribution /= np.sum(joint_distribution)

    return entropy(joint_distribution)


@numba.jit
def layer_information(layer_output, bins, py, px, unique_inverse_x, unique_inverse_y):
    tmp = np.digitize(layer_output, bins)
    ws_epoch_layer_bins = bins[tmp - 1]
    ws_epoch_layer_bins = ws_epoch_layer_bins.reshape(len(layer_output), -1)

    unique_t, unique_inverse_t, unique_counts_t = np.unique(
        ws_epoch_layer_bins, axis=0,
        return_index=False, return_inverse=True, return_counts=True
    )

    pt = unique_counts_t / np.sum(unique_counts_t)

    # # I(X, Y) = H(Y) - H(Y|X)
    # # H(Y|X) = H(X, Y) - H(X)

    x_entropy = entropy(px)
    y_entropy = entropy(py)
    t_entropy = entropy(pt)

    x_t_joint_entropy = joint_entropy(unique_inverse_x, unique_inverse_t, px.shape[0], layer_output.shape[0])
    y_t_joint_entropy = joint_entropy(unique_inverse_y, unique_inverse_t, py.shape[0], layer_output.shape[0])

    return {
        'local_IXT': t_entropy + x_entropy - x_t_joint_entropy,
        'local_ITY': y_entropy + t_entropy - y_t_joint_entropy
    }


@numba.jit
def calc_information_for_epoch(epoch_number, ws_epoch, bins, unique_inverse_x,
                               unique_inverse_y, pxs, pys):
    """Calculate the information for all the layers for specific epoch"""
    information_epoch = []

    for i in range(len(ws_epoch)):
        information_epoch_layer = layer_information(
            layer_output=ws_epoch[i],
            bins=bins,
            unique_inverse_x=unique_inverse_x,
            unique_inverse_y=unique_inverse_y,
            px=pxs, py=pys
        )
        information_epoch.append(information_epoch_layer)
    information_epoch = np.array(information_epoch)

    # print('Processed epoch {}'.format(epoch_number))

    return information_epoch


@numba.jit
def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    pys = np.sum(label, axis=0) / float(label.shape[0])

    unique_x, unique_x_indices, unique_inverse_x, unique_x_counts = np.unique(
        x, axis=0,
        return_index=True, return_inverse=True, return_counts=True
    )

    pxs = unique_x_counts / np.sum(unique_x_counts)

    unique_array_y, unique_y_indices, unique_inverse_y, unique_y_counts = np.unique(
        label, axis=0,
        return_index=True, return_inverse=True, return_counts=True
    )
    return pys, None, unique_x, unique_inverse_x, unique_inverse_y, pxs


def get_information(ws, x, label, num_of_bins, every_n=1,
                    return_matrices=False):
    """
    Calculate the information for the network for all the epochs and all the layers

    ws.shape =  [n_epoch, n_layers, n_params]
    ws --- outputs of all layers for all epochs
    """

    # print('Start calculating the information...')

    bins = np.linspace(-1, 1, num_of_bins)
    label = np.array(label).astype(np.float)
    pys, _, unique_x, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, x)

    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        information_total = parallel(
            delayed(calc_information_for_epoch)(
                i, epoch_output, bins, unique_inverse_x, unique_inverse_y, pxs, pys
            ) for i, epoch_output in enumerate(ws) if i % every_n == 0
        )

    if not return_matrices:
        return information_total

    ixt_matrix = np.zeros((len(information_total), len(ws[0])))
    ity_matrix = np.zeros((len(information_total), len(ws[0])))

    for epoch, layer_info in enumerate(information_total):
        for layer, info in enumerate(layer_info):
            ixt_matrix[epoch][layer] = info['local_IXT']
            ity_matrix[epoch][layer] = info['local_ITY']

    return ixt_matrix, ity_matrix

def plot_information_plane(IXT_array, ITY_array, num_epochs, every_n):
    assert len(IXT_array) == len(ITY_array)

    max_index = len(IXT_array)

    plt.figure(figsize=(15, 9))
    plt.xlabel('$I(X;T)$')
    plt.ylabel('$I(T;Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs + 1)]

    for i in range(0, max_index):
        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]
        plt.plot(IXT, ITY, marker='o', markersize=15, markeredgewidth=0.04,
                 linestyle=None, linewidth=1, color=colors[i * every_n], zorder=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Num epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(num_epochs), transform=cbar.ax.transAxes, va='bottom', ha='center')
    plt.savefig('tishby_plot_orig_4e-4lr.png')
    plt.show()


if __name__ == '__main__':
    #X, y = pytorch_network.load_tishby_toy_dataset('./data/g1.mat')
    X, y =  np.load("comparison/tishby_X.npy"), np.load("comparison/tishby_y.npy")
    from tishby_check import train_network, NNet, get_aligned_representations

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9)

    model = NNet()
    tracker = ActivationTracker(model.net, store_MI=False)
    train_res = train_network(model, X_train, y_train, X_test, y_test, tracker=tracker, batch_size=64, epochs=2000)

    q = [[e['tanh1'], e['tanh2'], e['tanh3'], e['tanh4'], e['tanh5']] for e in tracker.epoch_activations]
    order = train_res[2]

    ws = get_aligned_representations(q, order)

    import dill
    dill.dump(ws, file=open('comparison/activation_track.pickle', 'wb'))

    #ws = dill.load(open('activation_track_4e-5lr.pickle', 'rb'))
    num_of_bins = 40
    every_n = 10
    IXT_array, ITY_array = get_information(ws, X_train, np.concatenate([y_train, 1 - y_train], axis=1),
                                           num_of_bins, every_n=every_n, return_matrices=True)


    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({'font.size': 18})




    plot_information_plane(IXT_array, ITY_array, num_epochs=2000, every_n=every_n)