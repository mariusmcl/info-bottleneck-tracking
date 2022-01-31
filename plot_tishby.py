import argparse
import dill
import numpy as np
import matplotlib.pyplot as plt


def plot_train_and_val_accuracies(train_accs, val_accs, PATH, model_name, dataset_name):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.98, top=0.98, wspace=0, hspace=0)

    ax.plot(train_accs, color='black',
            #label=f'{model_name} training ({dataset_name}), final accuracy: {train_accs[-1]:.2f}')
            label=f"Train set final accuracy: {train_accs[-1]:.2f}")
    ax.plot(val_accs, color='black', linestyle='dashed',
            #label=f'{model_name} validation ({dataset_name}), final accuracy: {val_accs[-1]:.2f}')
            label=f"Validation set final accuracy: {val_accs[-1]:.2f}")


    ax.legend(loc='best')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Accuracy')
    plt.savefig(PATH + 'Accuracy.png')
    plt.show()

def plot_information_plane_paper(IXT_array, ITY_array, num_epochs, every_n, PATH,
                           xlim=None, ylim=None, inset=False):
    assert len(IXT_array) == len(ITY_array)
    scale = 6
    SMALL_SIZE = 16 
    MEDIUM_SIZE = SMALL_SIZE + 4 + scale
    BIGGER_SIZE = MEDIUM_SIZE + 4 + scale + 2

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=0.12, right=1.05, top=0.94, wspace=0, hspace=0)
    ax.set_xlabel('$I(X;Z)$')
    ax.set_ylabel('$I(Z;Y)$')
    ax.xaxis.label.set_size(SMALL_SIZE)
    ax.yaxis.label.set_size(SMALL_SIZE)
    ax.tick_params(labelsize=SMALL_SIZE)
    ax.tick_params(labelsize=SMALL_SIZE)
    max_index = len(IXT_array)
    # Create an inset in the lower right corner (loc=4) with borderpad=1, i.e.
    # 10 points padding (as 10pt is the default fontsize) to the parent axes

#    axins = inset_axes(ax, width="40%", height="40%", loc=4, borderpad=1)
    if inset:

        axins = ax.inset_axes([0.6, 0.08, 0.4, 0.3])
    #    axins.set_xlim(6.6, 7.2)
        if inset == "GCN":
            axins.set_xlim(14.14, 14.2)
            axins.set_ylim(4.43, 4.44)
        elif inset == "GAT":
            axins.set_xlim(14.05, 14.2)
            axins.set_ylim(4.4, 4.5)
        elif inset == "MLP":
            axins.set_xlim(13.8, 14.3)
            axins.set_ylim(4.3, 4.5)
        elif inset == "MLP-2":
            axins.set_xlim(4.806, 4.808)
            axins.set_ylim(2.421, 2.422)
        elif inset == "MLP-tanh-arxiv":
            axins.set_xlim(13.8, 14.2)
            axins.set_ylim(4.3, 4.5)
        elif inset == "GCN-tanh-arxiv":
            axins.set_xlim(13.8, 14.2)
            axins.set_ylim(4.35, 4.45)


        axins.tick_params(labelsize=SMALL_SIZE)
#    axins.xaxis.label.set_size(SMALL_SIZE)
#    axins.yaxis.label.set_size(SMALL_SIZE)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs + 1)]
    markers = ['o', 'v', 's', '^', 'X', 'P']

    for i in range(0, max_index):
        print("epoch:", i)
        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]
        print(IXT)
        print(ITY)
        for k in range(len(IXT)):
            ax.plot(IXT[k], ITY[k], marker=markers[k], markersize=12, markeredgewidth=0.04,
                 linestyle=None, linewidth=1, color=colors[i * every_n], zorder=10, alpha=0.5)
            if inset:
                axins.plot(IXT[k], ITY[k], marker=markers[k], markersize=12, markeredgewidth=0.04,
                    linestyle=None, linewidth=1, color=colors[i * every_n], zorder=10, alpha=0.5)

        for j in range(len(IXT)):
            #plt.plot(IXT, ITY, linestyle=None, linewidth=1, color=colors[i*every_n], zorder=10)
            ax.annotate(j+1, (IXT[j], ITY[j]), size=SMALL_SIZE-4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Training epochs', size=SMALL_SIZE)
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center', size=SMALL_SIZE)
    cbar.ax.text(0.5, 1.0, max_index, transform=cbar.ax.transAxes, va='bottom', ha='center', size=SMALL_SIZE)  # str(num_epochs)
    plt.savefig(PATH + "MIPlanev2" '.png')
    plt.show()

#    plt.savefig(figname + '.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Where the storage files are locatde")
    parser.add_argument("--directory", default="results/GCN-Cora-30122021-230616")
    parser.add_argument("--everyN", default=10, type=int, help="At what interval to plot the MI values")
    parser.add_argument("--inset", default="", type=str, help="Whether to plot with a custom inset")

    args = parser.parse_args()
    
    print("Directory:", args.directory, "everyN:", args.everyN)
    with open(args.directory + "/data.pickle", "rb") as f:
        data = dill.load(f)
    
    # Assume we have computed the MI-values during training:

    MI_values = data["storeMI"]  #  [{act1: [x, y], act2: [x, y]}, {act1: [x, y], act2: [x, y]}, .... ] 

    num_epochs, num_representations = len(MI_values), len(MI_values[0].keys())
    IXT_array = np.zeros(shape=(num_epochs, num_representations))
    ITY_array = np.zeros(shape=(num_epochs, num_representations))

    print("IXT shape:", IXT_array.shape)
    for epoch, info_dict in enumerate(MI_values):
        for i in range(1, num_representations + 1):
            IXT, ITY = info_dict["relu" + str(i)]
            IXT_array[epoch, i-1], ITY_array[epoch, i-1] = IXT, ITY

    PATH = args.directory + "/"
    val_accs, train_accs = data["val_accs"], data["train_accs"]
    items = PATH.split("-")
    model_name, dataset_name = items[0], items[1]

    plot_train_and_val_accuracies(train_accs=train_accs,val_accs=val_accs,PATH=PATH,model_name=model_name,dataset_name=dataset_name)

    plot_information_plane_paper(
        IXT_array=IXT_array,
        ITY_array=ITY_array,
        num_epochs=num_epochs,
        every_n=1,
        PATH=PATH,
        inset=args.inset
    )