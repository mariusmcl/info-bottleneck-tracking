from kolchinsky_MI import KOLCHINSKY_MUTUAL_INFO_COMPUTATION, plot_information_plane
from debug_tishby import IBSGD_ATTEMPT_MI
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import dill
from tracking_utils import DataTracker, GCNDataTracker



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.relu3 = nn.ReLU()
        self.conv4 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.relu1(x)
        #x = F.dropout(x, p=.5, training=self.training)
        #x = F.dropout(x, p=.5, training=self.training)
        #x = F.dropout(x, p=.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.relu2(x)

        x = self.conv3(x, edge_index)
        x = self.relu3(x)

        x = self.conv4(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_features):
        super(MLP, self).__init__()
        torch.manual_seed(12345)
        self.fc1 = nn.Linear(num_features, hidden_channels)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        return x


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, GCNConv):
        nn.init.zeros_(m.bias)
        nn.init.normal_(m.weight, mean=0.0, std=1. / np.sqrt(m.weight.data.shape[0]))


def evaluate_MLP(model, data_):
    model.eval()
    out = model(data_.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data_.test_mask] == data_.y[data_.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data_.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


def batch_train_MLP(model, tracker, optimizer, criterion, data_, num_epochs, batch_size):
    losses = []
    val_accuracies = []
    dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_.x, data_.y),
                                             batch_size=batch_size, shuffle=True)
    for epoch in range(1, num_epochs):
        tracker.register_new_epoch(['relu1', 'relu2', 'relu3', 'fc1', 'fc2', 'fc3'], ['fc1', 'fc2', 'fc3', 'fc4'])
        model.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().item())
            tracker.save()
        valacc = evaluate_MLP(model, data_)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valacc: {valacc:.2f}')#, end="\r")
        val_accuracies.append(valacc)
    return losses, val_accuracies


def fullbatch_train_MLP(model, tracker, optimizer, criterion, data_, num_epochs):
    losses = []
    val_accuracies = []
    train_accuracies = []
    for epoch in range(1, num_epochs):
        tracker.register_new_epoch(['relu1', 'relu2', 'relu3', 'fc1', 'fc2', 'fc3'], ['fc1', 'fc2', 'fc3', 'fc4'])
        model.train()
        optimizer.zero_grad()
        out = model(data_.x)
        loss = criterion(out[data_.train_mask], data_.y[data_.train_mask])
        loss.backward()
        tracker.save()
        losses.append(loss.cpu().item())
        optimizer.step()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct = pred[data_.train_mask] == data.y[data_.train_mask]
            acc = int(correct.sum()) / pred[data_.train_mask].shape[0]
            train_accuracies.append(acc)
        valacc = evaluate_MLP(model, data_)
        val_accuracies.append(valacc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valacc{valacc:.4f}, Trainacc {acc:.4f}')
    return losses, val_accuracies, train_accuracies








def make_grad_plot(ax, tracker_means, tracker_stds):
    plt.figure(figsize=(8, 6))
    num_layers = len(tracker_means)
    colors = ['black', 'blue', 'red', 'green', 'purple', 'orange']
    for i, layer in enumerate(tracker_means[0].keys()):   # loop over the names of the layers, mean_snr[0] is a dictinoary from layer -> meannorm
        x = np.arange(1, len(tracker_means) + 1)
        curr_layer_mean_norms = [element[layer] for element in tracker_means]   # element is now each epoch, extract the layers meannorm
        curr_layer_std_norms = [element[layer] for element in tracker_stds]
        ax.plot(x, curr_layer_mean_norms, linestyle="-", linewidth=1, color=colors[i],
                 label=r'$||Mean(\nabla W_{' + layer + '})||$')
        ax.plot(x, curr_layer_std_norms, linestyle="--", linewidth=1, color=colors[i],
                 label=r'$||Std(\nabla W_{' + layer + '})||$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    return ax



@torch.no_grad()
def evaluate_GCN(model, data_):
    model.eval()
    out = model(data_.x, data_.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[data_.val_mask] == data_.y[data_.val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(data_.val_mask.sum())  # Derive ratio of correct predictions.
    return val_acc


def train_GCN(model, tracker, optimizer, criterion, data_, num_epochs):
    losses, val_accuracies, train_accuracies = [], [], []
    for epoch in range(1, num_epochs):
        tracker.register_new_epoch(['relu1', 'relu2', 'relu3'], ['conv1', 'conv2', 'conv3', 'conv4'])
        model.train()
        # for batch in loader:
        # batch_cuda = batch.to(device)
        # adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(data_.x, data_.edge_index)
        loss = criterion(out[data_.train_mask], data_.y[data_.train_mask])
        loss.backward()
        tracker.save()
        losses.append(loss.cpu().item())
        optimizer.step()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct = pred[data_.train_mask] == data.y[data_.train_mask]
            acc = int(correct.sum()) / pred[data_.train_mask].shape[0]
            train_accuracies.append(acc)
        valacc = evaluate_GCN(model, data_)
        val_accuracies.append(valacc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valacc{valacc:.4f}, Trainacc {acc:.4f}')#, end="\r")
    return losses, val_accuracies, train_accuracies


def make_grad_plot_rolling(ax, tracker_means):
    # assumes 4x length on conv epochs than mlp epochs
    import pandas as pd
    plt.figure(figsize=(8, 6))
    colors = ['black', 'blue', 'red', 'green', 'purple', 'orange']
    for i, layer in enumerate(tracker_means[0].keys()):   # loop over the names of the layers, mean_snr[0] is a dictinoary from layer -> meannorm
        x = np.arange(1, len(tracker_means)//4 + 1)
        curr_layer_mean_norms = np.array([element[layer] for element in tracker_means])   # element is now each epoch, extract the layers meannorm
        curr_layer_std_norms = pd.Series(curr_layer_mean_norms).rolling(4).std()[4::4]
        curr_layer_mean_norms = pd.Series(curr_layer_mean_norms).rolling(4).mean()[4::4]
        ax.plot(x, curr_layer_mean_norms, linestyle="-", linewidth=1, color=colors[i],
                 label=r'$||Mean(\nabla W_{' + layer + '})||$')
        ax.plot(x, curr_layer_std_norms, linestyle="--", linewidth=1, color=colors[i],
                 label=r'$||Std(\nabla W_{' + layer + '})||$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    return ax



if __name__ == '__main__':
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

    device = 'cpu'
    data = dataset[0]
    torch.manual_seed(1)



    BATCH_SIZE = 256

    max_mlp_epochs = 601
    max_conv_epochs = 601

    do_train_MLP = False
    fullbatch_MLP = True

    do_train_GCN = False

    save_MLP = True
    save_GCN = True
    conv_LR = 1e-3
    mlp_LR = 1e-3

    EXPERIMENT_NAME = 'GRAPHLIN600EPOCHS_LOWERBOUND###2'

    if do_train_MLP:
        MLP_model = MLP(16, num_classes=dataset.num_classes, num_features=dataset.num_features)
        MLP_model.apply(initialize_weights)
        MLP_model.to(device)
        print(MLP_model)
        mlp_tracker = DataTracker(MLP_model, backward=True)

        if fullbatch_MLP:
            mlp_losses, mlp_val_acc, mlp_train_acc = fullbatch_train_MLP(model=MLP_model,
                                                          tracker=mlp_tracker,
                                                          optimizer=torch.optim.Adam(MLP_model.parameters(),
                                                                                     lr=mlp_LR),#, weight_decay=5e-4),
                                                          criterion=torch.nn.CrossEntropyLoss(),
                                                          data_=data, num_epochs=max_mlp_epochs)
        else:
            mlp_losses, mlp_val_acc = batch_train_MLP(model=MLP_model,
                                                      tracker=mlp_tracker,
                                                      optimizer=torch.optim.Adam(MLP_model.parameters(),
                                                                                 lr=mlp_LR),#, weight_decay=5e-4),
                                                      criterion=torch.nn.CrossEntropyLoss(),
                                                      data_=data, num_epochs=max_mlp_epochs, batch_size=BATCH_SIZE)

        # We have now tracked the activations and gradients for the linear model

        #fig, ax = plt.subplots()
        #ax = make_grad_plot(ax, mlp_tracker.grad_mean_snr_epochs, mlp_tracker.grad_std_snr_epochs)
        #plt.show()

        if save_MLP:
            dill.dump((mlp_tracker.epoch_activations, mlp_val_acc, mlp_train_acc), file=open(EXPERIMENT_NAME + '_LINACT.pickle', 'wb'))

            #dill.dump(mlp_tracker.gradients, file=open('data_dump/mlp_tracker_cora_gradients_' + 'linact_' + 'fullbatch'
            #                                                   + str(fullbatch_MLP) + '.pickle', 'wb'))

    if do_train_GCN:
        CONV_model = GCN(16, num_classes=dataset.num_classes, num_features=dataset.num_features)
        CONV_model.apply(initialize_weights)
        CONV_model.to(device)
        print(CONV_model)

        conv_tracker = GCNDataTracker(CONV_model, backward=True)

        gcn_losses, gcn_val_acc, gcn_train_acc = train_GCN(model=CONV_model,
                                            tracker=conv_tracker,
                                            optimizer=torch.optim.Adam(CONV_model.parameters(),
                                                                       lr=conv_LR),#, weight_decay=5e-4),
                                            criterion=torch.nn.CrossEntropyLoss(),
                                            data_=data, num_epochs=max_conv_epochs)

        #fig_conv, ax_conv = plt.subplots()
        #ax_conv = make_grad_plot_rolling(ax_conv, conv_tracker.grad_mean_snr_epochs)
        #plt.show()

        if save_GCN:
            dill.dump((conv_tracker.epoch_activations, gcn_val_acc, gcn_train_acc), file=open(EXPERIMENT_NAME + '_GCNACT.pickle', 'wb'))
            #dill.dump(conv_tracker.gradients, file=open('data_dump/conv_tracker_cora_gradients.pickle', 'wb'))


    (conv_activations, gcn_validation_accuracies, gcn_training_accuracies) = dill.load(open(EXPERIMENT_NAME + '_GCNACT.pickle', 'rb'))

    (mlp_activations, mlp_validation_accuracies, mlp_training_accuracies) = dill.load(open(EXPERIMENT_NAME + '_LINACT.pickle', 'rb'))
    #mlp_activations = dill.load(open(EXPERIMENT_NAME + '_LINACT.pickle', 'rb'))

    PLOT_ACCURACIES = True
    if PLOT_ACCURACIES:
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


        plt.clf()
        plt.plot(gcn_validation_accuracies, 'r', label=f'ValAcc: {gcn_validation_accuracies[-1]:.2f}')
        plt.plot(gcn_training_accuracies, 'b', label=f'TrainAcc: {gcn_training_accuracies[-1]:.2f}')
        plt.legend(loc='best')
        plt.xlabel('Epoch')
        plt.title(f'Graph-Convolutional', fontdict = {'fontsize' : BIGGER_SIZE})
        plt.ylabel('Accuracy')
        plt.savefig(EXPERIMENT_NAME + '_GCN_VALACC.png')
        plt.tight_layout()
        plt.show()

        plt.clf()
        plt.plot(mlp_validation_accuracies, 'r', label=f'ValAcc: {mlp_validation_accuracies[-1]:.2f}')
        plt.plot(mlp_training_accuracies, 'b', label=f'TrainAcc: {mlp_training_accuracies[-1]:.2f}')
        plt.legend(loc='best')
        plt.xlabel('Epoch')
        plt.title(f'Fully-Connected', fontdict = {'fontsize' : BIGGER_SIZE})
        plt.ylabel('Accuracy')
        plt.savefig(EXPERIMENT_NAME + '_MLP_VALACC.png')
        plt.tight_layout()
        plt.show()

    conv_storage = {'IX1': [], 'IX2': [], 'IX3': [],
                    'I1Y': [], 'I2Y': [], 'I3Y': []}
    mlp_storage = {'IX1': [], 'IX2': [], 'IX3': [],
                    'I1Y': [], 'I2Y': [], 'I3Y': []}

    SAVE_MI = True
    EXPERIMENT_NAME = EXPERIMENT_NAME + '_VARIABLENOISE#2'   # i tilfelle man vil endre noise-parameter
    if SAVE_MI:
        for epoch, (conv_epoch, mlp_epoch) in enumerate(zip(conv_activations, mlp_activations)):
            for i in [1, 2, 3]:
                conv_z = conv_epoch['relu' + str(i)]
                mlp_z = mlp_epoch['relu' + str(i)]

                Z, Y = conv_z[data.train_mask], data.y[data.train_mask].unsqueeze(1)
                #ixz_conv, izy_conv = IBSGD_ATTEMPT_MI(Z, Y, 0.001) #KOLCHINSKY_MUTUAL_INFO_COMPUTATION(Z, Y, noise_variance=0.1)
                noisedict = {1: 1e-3, 2: 3e-3, 3: 1e-2}
                noise = noisedict[i]
                ixz_conv, izy_conv = KOLCHINSKY_MUTUAL_INFO_COMPUTATION(Z, Y, noise)
                Z = mlp_z[data.train_mask]
                #ixz_mlp, izy_mlp = IBSGD_ATTEMPT_MI(Z, Y, 0.001) #KOLCHINSKY_MUTUAL_INFO_COMPUTATION(Z, Y, noise_variance=0.1)
                ixz_mlp, izy_mlp = KOLCHINSKY_MUTUAL_INFO_COMPUTATION(Z, Y, noise_variance=noise)
                # venter litt med mlp beregningen
                conv_storage['IX' + str(i)].append(ixz_conv)
                conv_storage['I' + str(i) + 'Y'].append(izy_conv)
                mlp_storage['IX' + str(i)].append(ixz_mlp)
                mlp_storage['I' + str(i) + 'Y'].append(izy_mlp)
            print(f'Epoch: {epoch:03d}', end="\r")
        dill.dump(conv_storage, file=open(EXPERIMENT_NAME + '_GCN_MI.pickle', 'wb'))
        dill.dump(mlp_storage, file=open(EXPERIMENT_NAME + 'LIN_MI.pickle', 'wb'))
    else:
        conv_storage = dill.load(open(EXPERIMENT_NAME + '_GCN_MI.pickle', 'rb'))
        mlp_storage = dill.load(open(EXPERIMENT_NAME + 'LIN_MI.pickle', 'rb'))


    conv_IXT_array = np.hstack(
        (np.expand_dims(conv_storage['IX1'], axis=1), np.expand_dims(conv_storage['IX2'], axis=1),
         np.expand_dims(conv_storage['IX3'], axis=1)))
    conv_ITY_array = np.hstack(
        (np.expand_dims(conv_storage['I1Y'], axis=1), np.expand_dims(conv_storage['I2Y'], axis=1),
         np.expand_dims(conv_storage['I3Y'], axis=1)))

    mlp_IXT_array = np.hstack(
        (np.expand_dims(mlp_storage['IX1'], axis=1), np.expand_dims(mlp_storage['IX2'], axis=1),
         np.expand_dims(mlp_storage['IX3'], axis=1)))
    mlp_ITY_array = np.hstack(
        (np.expand_dims(mlp_storage['I1Y'], axis=1), np.expand_dims(mlp_storage['I2Y'], axis=1),
         np.expand_dims(mlp_storage['I3Y'], axis=1)))

    PLOT_MIS = True
    if PLOT_MIS:
        plot_information_plane(conv_IXT_array[::10], conv_ITY_array[::10], max_conv_epochs - 1, 10, figname=EXPERIMENT_NAME + 'GCN_INFO_PLANE',
                               title=f'Graph-Convolutional')

        plot_information_plane(mlp_IXT_array[::10], mlp_ITY_array[::10], max_mlp_epochs - 1, 10, figname=EXPERIMENT_NAME + 'LINEAR_INFO_PLANE',
                               title=f'Fully-Connected')#, xlim=(7, 7.15), ylim=(2.6, 3))

    # (final) mlp valacc: 0.44, mlp_trainacc: ~0.8 ( trent på 140 datapunkter, validert på 1000)
    # (final) conv valacc: 0.74, mlp_trainacc ~0.96 ( samme som over) --> variability i trainacc pga få datapkt
    a = 2

