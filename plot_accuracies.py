import matplotlib.pyplot as plt
from numpy import mod
import numpy as np
import dill

def plot_train_and_val_accuracies(gcn_train_accs, gcn_val_accs, mlp_train_accs, mlp_val_accs, PATH):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.12, bottom=0.14, right=0.98, top=0.98, wspace=0, hspace=0)

    ax.plot(gcn_train_accs, color='black',
            #label=f'{model_name} training ({dataset_name}), final accuracy: {train_accs[-1]:.2f}')
            label=f"GCN training, final accuracy: {gcn_train_accs[-1]:.2f}")
    ax.plot(gcn_val_accs, color='black', linestyle='dashed',
            #label=f'{model_name} validation ({dataset_name}), final accuracy: {val_accs[-1]:.2f}')
            label=f"GCN validation, final accuracy: {gcn_val_accs[-1]:.2f}")

    ax.plot(mlp_train_accs, 'blue',
                label=f'MLP training, final accuracy: {mlp_train_accs[-1]:.2f}')
    ax.plot(mlp_val_accs, 'blue', linestyle='dashed',
                label=f'MLP validation, final accuracy: {mlp_val_accs[-1]:.2f}')

    ax.legend(loc='lower right')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(PATH + 'Accuracy.png')
    plt.show()


if __name__ == "__main__":

    MLP_SOURCE = "MLP-Cora-100E-30012022-124835-RELU-UPPERBOUND"

    GCN_SOURCE = "GCN-Cora-100E-30012022-124456-RELU-UPPERBOUND"

    with open("results/" + MLP_SOURCE + "/data.pickle", "rb") as f:
        mlp_data = dill.load(f)
    

    with open("results/" + GCN_SOURCE + "/data.pickle", "rb") as f:
        gcn_data = dill.load(f)
    
    PATH = "results" + "/"
    mlp_val_accs, mlp_train_accs = mlp_data["val_accs"], mlp_data["train_accs"]
    gcn_val_accs, gcn_train_accs = gcn_data["val_accs"], gcn_data["train_accs"]

    plot_train_and_val_accuracies(
        mlp_train_accs=mlp_train_accs,
        mlp_val_accs=mlp_val_accs,
        gcn_train_accs=gcn_train_accs,
        gcn_val_accs=gcn_val_accs,
        PATH=PATH
    )