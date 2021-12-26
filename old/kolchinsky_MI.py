import dill
from sklearn.model_selection import train_test_split
import dataset_utils
import torch
import numpy as np
import math
import matplotlib.pyplot as plt


def compute_distances(x):
    '''
    Computes the distance matrix for the KDE Entropy estimation:
    - x (Tensor) : array of functions to compute the distances matrix from
    '''

    x_norm = (x**2).sum(1).view(-1,1)
    x_t = torch.transpose(x,0,1)
    x_t_norm = x_norm.view(1,-1)
    dist = x_norm + x_t_norm - 2.0*torch.mm(x,x_t)
    dist = torch.clamp(dist,0,np.inf)

    return dist

def KDE_IXT_estimation(mean_t):
    '''
    Computes the MI estimation of X and T. Parameters:
    - logvar_t (float) : log(var) of the bottleneck variable
    - mean_t (Tensor) : deterministic transformation of the input
    '''

    n_batch, d = mean_t.shape
    var = 0.1

    # calculation of the constant
    normalization_constant = math.log(n_batch)

    # calculation of the elements contribution
    dist = compute_distances(mean_t)
    distance_contribution = - torch.mean(torch.logsumexp(input=- 0.5 * dist / var,dim=1))

    # mutual information calculation (natts)
    I_XT = normalization_constant + distance_contribution
    return I_XT

def KOLCHINSKY_MUTUAL_INFO_COMPUTATION_TWO(Z, Y):
    P = Z.shape[0]
    assert P == Y.shape[0]
    Z = torch.from_numpy(Z)
    input_mutual_information = KDE_IXT_estimation(Z)   # computes for X
    labels, counts = np.unique(Y, return_counts=True)

    s = 0
    for i, label in enumerate(labels):   # Denne er bare labels lang så den får gå
        label_activations = Z[(Y == label)[:, 0], :]
        assert label_activations.shape[0] == counts[i]
        label_output_dist_mtrx_logsumexp = compute_pairwise(label_activations, label_activations)
        s += (counts[i] / P) * label_output_dist_mtrx_logsumexp
    output_mutual_info = input_mutual_information - s

    return input_mutual_information, output_mutual_info


def KOLCHINSKY_MI_INPUT_Z(Z, X):
    # X trengs egentlig ikke her, men er bare for å huske at detter er infoen til X og Z
    P = X.shape[0]
    Z = torch.from_numpy(Z)
    sigma_squared = 0.1   # fra ontheinfobottleneck paper

    pairwise_dists = torch.cdist(Z.unsqueeze(0), Z.unsqueeze(0)).squeeze() ** 2

    pairwise_dists = torch.exp(-1/(2*sigma_squared) * pairwise_dists).sum(dim=1) / P

    pairwise_dists = -torch.mean(torch.log(pairwise_dists))


    return pairwise_dists.numpy()


def compute_pairwise(A, B, noise_variance=0.001):
    P = A.shape[0]
    H = A.shape[1]   # antas likt da.. (A.shape[1] = B.shape[1])
    #sigma_squared = 0.001   # fra ontheinfobottleneck paper

    """
       pairwise_dists = -torch.mean(torch.log(pairwise_dists))   # detete var det jeg returnerte tidligere


       dims, N = get_shape(x)
       dists = Kget_dists(x)
       dists2 = dists / (2 * var)
       normconst =

       lprobs = K.logsumexp(-dists2, axis=1) - K.log(N) - normconst
       h = -K.mean(lprobs) # + K.log(N) + normconst


       #return dims / 2 + h

       return pairwise_dists.numpy() + H/2 + np.log(P) + normconst

            #pairwise_dists = torch.exp(-1 / (2 * sigma_squared) * pairwise_dists).sum(dim=1) / P
    a, b, c = np.log(P), normconst, -torch.mean(pairwise_dists).numpy()
    normconst = (H / 2.0) * torch.log(torch.tensor(2 * np.pi * sigma_squared).float())

    """

    normconst = (H/2.0) * torch.log(torch.tensor(2 * np.pi * noise_variance, dtype=torch.float))

    pairwise_dists = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze() ** 2

    pairwise_dists = torch.logsumexp(-pairwise_dists / (2 * noise_variance), dim=1) - torch.log(torch.tensor(P, dtype=torch.float)) - normconst

    return-torch.mean(pairwise_dists) + H/2  #-torch.mean(pairwise_dists).numpy() + np.log(P) #+ normconst + H/2


def KOLCHINSKY_MUTUAL_INFO_COMPUTATION(Z, Y, noise_variance=0.001):
    P, H = Z.shape[0], Z.shape[1]
    assert P == Y.shape[0]
    Z = torch.from_numpy(Z)
    input_mutual_information = compute_pairwise(Z, Z, noise_variance=noise_variance)   # computes for X
    labels, counts = np.unique(Y, return_counts=True)

    s = 0.0
    for i, label in enumerate(labels):   # Denne er bare labels lang så den får gå
        """
        view = (Y == label).numpy()
        view_Y = Y.numpy()
        view_lables = label_activations.numpy()
        view_data = Z.numpy()
        
        """
        label_activations = Z[(Y == label)[:, 0], :]
        assert label_activations.shape[0] == counts[i]
        label_output_dist_mtrx_logsumexp = compute_pairwise(label_activations, label_activations, noise_variance=noise_variance)
        s += (counts[i] / P) * label_output_dist_mtrx_logsumexp
    output_mutual_info = input_mutual_information - s
    input_mutual_information = input_mutual_information - (H/2.0) * (torch.log(torch.tensor(2*np.pi*noise_variance, dtype=torch.float)) + 1.0)
    # forsøk på ny ITY computation:
    #HY_given_T = torch.nn.CrossEntropyLoss(Z, Y)
    #ITY = (np.log(2) - HY_given_T)
    return input_mutual_information / torch.log(torch.tensor(2.0, dtype=torch.float)), output_mutual_info / torch.log(torch.tensor(2.0, dtype=torch.float))


def plot_information_plane(IXT_array, ITY_array, num_epochs, every_n, figname='default', title='default', xlim=None, ylim=None):
    assert len(IXT_array) == len(ITY_array)
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
    #for j in range(3):
        #markertype = ['o', '^', 's']    # final layer er square, første er runding og 2. er trekant
    #    plt.plot(IXT_array[:, j], ITY_array[:, j], marker='o', markersize=15, markeredgewidth=0.04, linestyle='None')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Num epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, 70, transform=cbar.ax.transAxes, va='bottom', ha='center')  # str(num_epochs)
    plt.title(title, fontdict = {'fontsize' : BIGGER_SIZE})
    plt.tight_layout()
    plt.savefig(figname + '.png')
    plt.show()


if __name__ == 'main':

    ws = dill.load(open('data_dump/activation_track_linact_4e-4.pickle', 'rb'))  # get the activations

    X, y = dataset_utils.load_tishby_toy_dataset('./data/g1.mat')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9)

    mutual_info_dictionary = {'XZ1': [], 'XZ2': [], 'XZ3': [],
                              'XZ4': [], 'XZ5': [],
                              'Z1Y': [], 'Z2Y': [], 'Z3Y': [],
                              'Z4Y': [], 'Z5Y': []}
    print('starting minfo')
    for j, epoch in enumerate(ws):
        # Now we need to compute the mutual information for each layer wrt. X and Y
        #information_net_dictionary = get_dict()   # Start with a fresh model each epoch
        for i in [1, 2, 3, 4, 5]:
            xzi, ziy = KOLCHINSKY_MUTUAL_INFO_COMPUTATION(ws[j][i-1], y_train)
            mutual_info_dictionary['XZ'+ str(i)].append(xzi)
            mutual_info_dictionary['Z'+str(i)+'Y'].append(ziy)
        print(f'Epoch: {j:03d}')


    dill.dump(mutual_info_dictionary, file=open('data_dump/minfo_linact.pickle', 'wb'))

    #mutual_info_dictionary = dill.load(file=open('minfo.pickle', 'rb'))

    for key in mutual_info_dictionary:
        mutual_info_dictionary[key] = np.expand_dims(np.array(mutual_info_dictionary[key]), axis=1)
    mine_IXT_array = np.hstack((mutual_info_dictionary['XZ1'], mutual_info_dictionary['XZ2'], mutual_info_dictionary['XZ3'],
                           mutual_info_dictionary['XZ4'], mutual_info_dictionary['XZ5']))
    mine_ITY_array = np.hstack((mutual_info_dictionary['Z1Y'], mutual_info_dictionary['Z2Y'], mutual_info_dictionary['Z3Y'],
                           mutual_info_dictionary['Z4Y'], mutual_info_dictionary['Z5Y']))

    plot_information_plane(mine_IXT_array[::5], mine_ITY_array[::5], 2000, 5, figname='tishby_linact_Kolchinsky_4e-4')
