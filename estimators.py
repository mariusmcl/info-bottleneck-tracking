import dill
from sklearn.model_selection import train_test_split
import dataset_utils
import torch
import numpy as np
import math
import matplotlib.pyplot as plt


def compute_pairwise(A, B, noise_variance=0.001):
    P = A.shape[0]
    H = A.shape[1]   # they should be (A.shape[1] = B.shape[1])

    assert A.shape[1] == B.shape[1]

    normconst = (H/2.0) * torch.log(torch.tensor(2 * np.pi * noise_variance, dtype=torch.float))

    pairwise_dists = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze() ** 2

    pairwise_dists = torch.logsumexp(-pairwise_dists / (2 * noise_variance), dim=1) - torch.log(torch.tensor(P, dtype=torch.float)) - normconst

    return -torch.mean(pairwise_dists) + H/2 



def KOLCHINSKY_MUTUAL_INFO_COMPUTATION(Z, Y, noise_variance=0.001):
    P, H = Z.shape[0], Z.shape[1]
    assert P == Y.shape[0]
    Z = torch.from_numpy(Z)
    input_mutual_information = compute_pairwise(Z, Z, noise_variance=noise_variance)   # computes for X
    labels, counts = np.unique(Y, return_counts=True)

    s = 0.0
    
    for i, label in enumerate(labels):   # Denne er bare labels lang så den får gå
        label_activations = Z[(Y == label)[:, 0], :]
        assert label_activations.shape[0] == counts[i]
        label_output_dist_mtrx_logsumexp = compute_pairwise(label_activations, label_activations, noise_variance=noise_variance)
        s += (counts[i] / P) * label_output_dist_mtrx_logsumexp

    output_mutual_info = input_mutual_information - s

    input_mutual_information = input_mutual_information - (H/2.0) * (torch.log(torch.tensor(2*np.pi*noise_variance, dtype=torch.float)) + 1.0)
    return input_mutual_information / torch.log(torch.tensor(2.0, dtype=torch.float)), output_mutual_info / torch.log(torch.tensor(2.0, dtype=torch.float))
