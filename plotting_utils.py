import matplotlib.pyplot as plt
from numpy import mod
import numpy as np


def plot_train_and_val_accuracies(train_accs, val_accs, PATH, model_name, dataset_name):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.98, top=0.98, wspace=0, hspace=0)

    ax.plot(train_accs, color='black',
            label=f'{model_name} training ({dataset_name}), final accuracy: {train_accs[-1]:.2f}')
    ax.plot(val_accs, color='black', linestyle='dashed',
            label=f'{model_name} validation ({dataset_name}), final accuracy: {val_accs[-1]:.2f}')

    # ax.plot(mlp_training_accuracies, 'blue',
    #             label=f'MLP training, final accuracy: {mlp_training_accuracies[-1]:.2f}')
    # ax.plot(mlp_validation_accuracies, 'blue', linestyle='dashed',
    #             label=f'MLP validation, final accuracy: {mlp_validation_accuracies[-1]:.2f}')

    ax.legend(loc='best')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Accuracy')
    plt.savefig(PATH + 'Accuracy.png')
    plt.show()

def plot_information_plane(IXT_array, ITY_array, num_epochs, every_n,
                           model_name, dataset_name, PATH, xlim=None, ylim=None):

    IXT_array, ITY_array = IXT_array[::every_n], ITY_array[::every_n]
    scale = 6
    SMALL_SIZE = 20 + scale
    MEDIUM_SIZE = 24 + scale
    BIGGER_SIZE = 28 + scale + 2

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    max_index = len(IXT_array)

    plt.figure(figsize=(15, 9))
    plt.xlabel('$I(X;T)$')
    plt.ylabel('$I(T;Y)$')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    #num_epochs = 70
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs + 1)]

    for i in range(0, max_index):
        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]
        plt.plot(IXT, ITY, marker='o', markersize=12, markeredgewidth=0.04,
                 linestyle=None, linewidth=1, color=colors[i * every_n], zorder=10)
        #plt.plot(IXT, ITY, linestyle=None, linewidth=1, color=colors[i*every_n], zorder=10)
        for j in range(len(IXT)):
            plt.annotate(j+1, (IXT[j], ITY[j]), size=SMALL_SIZE)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Num epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, num_epochs, transform=cbar.ax.transAxes, va='bottom', ha='center')  # str(num_epochs)
    plt.tight_layout()
    plt.savefig(PATH + "MIPlane" '.png')
    plt.show()