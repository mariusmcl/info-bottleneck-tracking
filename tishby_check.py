import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
#from tracking_utils import DataTracker
#from kolchinsky_MI import KOLCHINSKY_MUTUAL_INFO_COMPUTATION, plot_information_plane
from activation_trackers import ActivationTracker
#import activation_trackers
from activation_trackers import ActivationTracker
from estimators import KOLCHINSKY_MUTUAL_INFO_COMPUTATION
from plotting_utils import plot_information_plane
import dill
import tqdm



class BatchGenerator():
    def __init__(self, inputs_list, batch_size, seed=None):
        self.inputs_list = inputs_list
        self.batch_size = batch_size
        self.seed = seed

        self.indices = np.arange(self.inputs_list[0].shape[0])
        np.random.seed(self.seed)
        np.random.shuffle(self.indices)

    def how_it_shuffled(self):
        return [current_input[self.indices] for current_input in self.inputs_list], np.arange(len(self.indices))[
            self.indices]

    def batch_generator(self):
        assert (len(self.inputs_list) > 0)

        for input_array in self.inputs_list:
            assert (input_array.shape[0] == self.inputs_list[0].shape[0])

        data_size = self.inputs_list[0].shape[0] // self.batch_size

        if self.inputs_list[0].shape[0] % self.batch_size > 0:
            data_size += 1

        for i in range(0, data_size):
            current_indices = self.indices[i * self.batch_size: (i + 1) * self.batch_size]
            yield [current_input[current_indices] for current_input in self.inputs_list]

def load_tishby_toy_dataset(filename, assign_random_labels=False, seed=42):
    np.random.seed(seed)

    data = sio.loadmat(filename)
    F = data['F']

    if assign_random_labels:
        y = np.random.randint(0, 2)
    else:
        y = data['y'].T

    return F, y

def get_aligned_representations(representations, order):
    for epoch in range(len(representations)):
        for layer in range(len(representations[0])):
            representations[epoch][layer] = representations[epoch][layer][np.argsort(order[epoch]), :]

    return representations


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(12, 10)), ('tanh1', nn.Tanh()),
            ('fc2', nn.Linear(10, 7)), ('tanh2', nn.Tanh()),
            ('fc3', nn.Linear(7, 5)), ('tanh3', nn.Tanh()),
            ('fc4', nn.Linear(5, 4)), ('tanh4', nn.Tanh()),
            ('fc5', nn.Linear(4, 3)), ('tanh5', nn.Tanh()),
            ('fc6', nn.Linear(3, 1))]))#, ('sigmoid', nn.Sigmoid())]))
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.normal_(m.weight, mean=0.0, std=1. / np.sqrt(m.weight.data.shape[0]))
        self.net.apply(weight_init)
    def forward(self, x):
        return self.net(x)



def fullbatch_train_MLP(model, tracker, optimizer, criterion, data_, num_epochs):
    losses = []
    val_accuracies = []
    for epoch in range(1, num_epochs):
        tracker.register_new_epoch(['tanh1', 'tanh2', 'tanh3', 'tanh4', 'tanh5'])
        model.train()
        optimizer.zero_grad()
        out = model(data_['X_train'])
        loss = criterion(out, data_['y_train'])
        loss.backward()
        tracker.save()
        losses.append(loss.cpu().item())
        optimizer.step()
        with torch.no_grad():
            pred = out.argmax(dim=1).unsqueeze(1)
            correct = pred == data['y_train']
            acc = int(correct.sum()) / pred.shape[0]
        valacc = evaluate_MLP(model, data_)
        val_accuracies.append(valacc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Valacc{valacc:.4f}, Trainacc {acc:.4f}')
    return losses, val_accuracies

def train_network(model, X, y, X_val, y_val, tracker=None, batch_size=12, epochs=16):
    """
    The network is trained with full batch
    """
    loss_list = []
    epoch_mean_loss = []
    accuracy_mean_val = []

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0004)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    loss_fun = nn.BCEWithLogitsLoss()
    #model.reset()
    train_shuffles = []

    for epoch in tqdm.tqdm(range(epochs)):
        samples = 0
        cum_loss = 0

        #model.reset()

        train_batcher = BatchGenerator([X, y], batch_size)
        train_shuffles.append(train_batcher.how_it_shuffled()[1])
        if tracker is not None:
            #tracker.register_new_epoch([str(i) for i in model.info_layers_numbers], ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6'])
            tracker.register_new_epoch(['tanh1', 'tanh2', 'tanh3', 'tanh4', 'tanh5'])
        for X_batch, y_batch in train_batcher.batch_generator():
            X_batch = torch.Tensor(X_batch)
            y_batch = torch.Tensor(y_batch)

            #model.train()
            predictions = model(X_batch)

            loss = loss_fun(predictions.reshape(-1), y_batch.reshape(-1))
            loss.backward()
            if tracker is not None:

                tracker.save()  # save on each training batch

            loss_list.append(loss.item())
            #print(loss.item())

            optimizer.step()
            optimizer.zero_grad()

            samples += X_batch.shape[0]
            cum_loss += loss.item()

        #scheduler.step()
        #model.next_epoch()

        epoch_mean_loss.append(cum_loss / samples)

        samples_val = 0
        accuracy_val = 0

        val_batcher = BatchGenerator([X_val, y_val], 1)

        for X_batch, y_batch in val_batcher.batch_generator():
            X_batch = torch.Tensor(X_batch)
            y_batch = torch.Tensor(y_batch)

            #model.eval()
            predictions_logits = model(X_batch)

            accuracy_val += (
                        y_batch.int() == (torch.nn.functional.sigmoid(predictions_logits) > 0.5).int()).sum().item()
            samples_val += X_batch.shape[0]

        accuracy_mean_val.append(float(accuracy_val) / samples_val)
        #print()
        #print(str(epoch) + ':', float(accuracy_val) / samples_val)
        print(f'Epoch: {epoch:03d}, Val_Acc: {float(accuracy_val) / samples_val:.4f}')#, end="\r")

    return epoch_mean_loss, accuracy_mean_val, train_shuffles, loss_list


def evaluate_MLP(model, data_):
    model.eval()
    out = model(data_['X_test'])
    pred = out.argmax(dim=1).unsqueeze(1)  # Use the class with highest probability.
    test_correct = pred == data_['y_test']  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / pred.shape[0]
    return test_acc


if __name__ == "__main__":


    #X, y = load_tishby_toy_dataset('./data/g1.mat')
    X, y =  np.load("comparison/tishby_X.npy"), np.load("comparison/tishby_y.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9)
    X_train, X_test = map(lambda x: torch.tensor(x, dtype=torch.float), (X_train, X_test))


    y_train, y_test = map(lambda x: torch.tensor(x, dtype=torch.float), (y_train, y_test))
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    model = NNet()

    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            nn.init.normal_(m.weight, mean=0.0, std=1. / np.sqrt(m.weight.data.shape[0]))

    model.apply(weight_init)

    print(model)


    optimizer = torch.optim.SGD(model.parameters(), lr=0.004)
    loss_fun = nn.BCEWithLogitsLoss()

    mlp_tracker = ActivationTracker(model.net, store_MI=False)

    NUM_EPOCHS = 200
    do_train = True
    if do_train:
        losses, val_accs = fullbatch_train_MLP(model=model, tracker=mlp_tracker, optimizer=optimizer,
                                            criterion=loss_fun, num_epochs=NUM_EPOCHS, data_=data)

        save_grads_and_activations = True
        if save_grads_and_activations:
            dill.dump(mlp_tracker.epoch_activations, file=open('mlp_tishby_activations.pickle', 'wb'))

    train_res = train_network(model, X_train, y_train, X_test, y_test, tracker=mlp_tracker, batch_size=64, epochs=NUM_EPOCHS)

    q = [[e['tanh1'], e['tanh2'], e['tanh3'], e['tanh4'], e['tanh5']] for e in mlp_tracker.epoch_activations]
    order = train_res[2]

    ws = get_aligned_representations(q, order)

    mlp_activations = ws

    #mlp_activations = dill.load(open('data_dump/mlp_tracker_tishby_activations.pickle', 'rb'))
    from information_process import get_information

    num_of_bins = 40
    every_n = 10
    mlp_IXT_array, mlp_ITY_array = get_information(ws, X_train, np.concatenate([y_train, 1 - y_train], axis=1),
                                        num_of_bins, every_n=every_n, return_matrices=True)

    plot_information_plane(mlp_IXT_array, mlp_IXT_array, NUM_EPOCHS - 1, 1,
                        figname='mlp_tishby' + str(0.004) + 'lr+0.1noise',
                        title=f'mlp_tishby')#, valacc {val_accs[-1]:.2f}')
    quit(0)

    mlp_storage = {'IX1': [], 'IX2': [], 'IX3': [], 'IX4': [], 'IX5': [],
                'I1Y': [], 'I2Y': [], 'I3Y': [], 'I4Y': [], 'I5Y': []}

    do_MI_COMPUTATION = False
    if do_MI_COMPUTATION:
        for epoch, mlp_epoch in enumerate(mlp_activations):
            for i in [1, 2, 3, 4, 5]:
            # mlp_z = mlp_epoch['tanh' + str(i)]
                mlp_z = mlp_epoch[i-1]
                Z, Y = mlp_z, data['y_train']
                ixz_mlp, izy_mlp = KOLCHINSKY_MUTUAL_INFO_COMPUTATION(Z, Y, noise_variance=0.1)

                mlp_storage['IX' + str(i)].append(ixz_mlp)
                mlp_storage['I' + str(i) + 'Y'].append(izy_mlp)
            print(f'Epoch: {epoch:03d}')#, end="\r")

    if do_MI_COMPUTATION:
        dill.dump(mlp_storage, file=open('data_dump/mlp_storage.pickle', 'wb'))
    else:
        mlp_storage = dill.load(open('data_dump/mlp_storage.pickle', 'rb'))

    mlp_IXT_array = np.hstack(
        (np.expand_dims(mlp_storage['IX1'], axis=1), np.expand_dims(mlp_storage['IX2'], axis=1),
        np.expand_dims(mlp_storage['IX3'], axis=1), np.expand_dims(mlp_storage['IX4'], axis=1),
        np.expand_dims(mlp_storage['IX5'], axis=1)))
    mlp_ITY_array = np.hstack(
        (np.expand_dims(mlp_storage['I1Y'], axis=1), np.expand_dims(mlp_storage['I2Y'], axis=1),
        np.expand_dims(mlp_storage['I3Y'], axis=1), np.expand_dims(mlp_storage['I4Y'], axis=1),
        np.expand_dims(mlp_storage['I5Y'], axis=1)))



    plot_information_plane(mlp_IXT_array, mlp_IXT_array, NUM_EPOCHS - 1, 1,
                        figname='mlp_tishby' + str(0.004) + 'lr+0.1noise',
                        title=f'mlp_tishby')#, valacc {val_accs[-1]:.2f}')

