from numpy.core.fromnumeric import prod
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import torch



def get_data_and_indices(dataset_name, reduction_factor=None):
    if dataset_name == "Cora":
        dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures()) 
        splits = {"train": dataset.data.train_mask, "valid": dataset.data.val_mask, "test": dataset.data.test_mask}

        model_params = {"num_classes":7, "num_features": 1433}
        return  dataset.data, splits, model_params

    elif dataset_name == "arxiv":
        arxiv_dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = 'data/arxiv')
        splits = arxiv_dataset.get_idx_split()

        arxiv_dataset.data.y = arxiv_dataset.data.y.squeeze()
        model_params = {"num_classes":40, "num_features": 128}
        return arxiv_dataset.data, splits, model_params

    elif dataset_name == "products":
        products_dataset = PygNodePropPredDataset(name = "ogbn-products", root = 'data/products')
        splits = products_dataset.get_idx_split()

        model_params = {"num_classes":47, "num_features":100}
        return products_dataset.data, splits, model_params
    
    elif dataset_name == "Tishby":
        X, y = np.load("comparison/tishby_X.npy"), np.load("comparison/tishby_y.npy")
        N = X.shape[0]
        num_val = N // 10
        train_idx = [True if i < (N - num_val) else False for i in range(N)]
        val_idx = [False if i < (N - num_val) else True for i in range(N)]
        print(f"Num train: f{len(train_idx)}, val: {len(val_idx)}, last train_idx/first_validx: {train_idx[-1]}, {val_idx[0]}")
        return torch.tensor(X).float(), torch.tensor(y).long(), {"train": torch.tensor(train_idx).bool(), "valid": torch.tensor(val_idx).bool()}
    else:
        print(f"Dataset {dataset_name} not found")


def clean_and_remove_indices(data, splits, dataset_name):
    if dataset_name == "arxiv":
        REDUCTION_FACTOR = 8 if reduction_factor is None else reduction_factor
        splits["train"] = splits["train"][::REDUCTION_FACTOR]   # Went out of memory (33GB) when comupting full
        splits["valid"] = splits["valid"][::REDUCTION_FACTOR]   
        
        idx_to_run = torch.cat((splits["train"], splits["valid"]), 0)
        ss = set(list(idx_to_run.numpy()))
        lookup_dict = {}
        for j in range(idx_to_run.shape[0]):
            value = int(idx_to_run[j])
            lookup_dict[value] = j   # Map the previous index to the current index
        new_edge_index = []

        print("Removing edges")
        for j in range(arxiv_dataset.data.edge_index.shape[1]):
            a, b = int(arxiv_dataset.data.edge_index[0, j].numpy()), int(arxiv_dataset.data.edge_index[1, j].numpy())
            if (a in ss) and (b in ss):
                assert (a in lookup_dict.keys()) and (b in lookup_dict.keys())
                a, b = lookup_dict[a], lookup_dict[b]
                new_edge_index.append([a, b])
        print("Finished removing edges")
        qq = np.array(new_edge_index)
        qq = torch.from_numpy(qq).type_as(arxiv_dataset.data.edge_index).reshape((2, -1)).contiguous()

        print("dataset shape before reduction:")
        print(arxiv_dataset.data.x.shape)
        print(arxiv_dataset.data.y.shape)
        print(arxiv_dataset.data.edge_index.shape)
        arxiv_dataset.data.x = arxiv_dataset.data.x[idx_to_run]
        arxiv_dataset.data.y = arxiv_dataset.data.y[idx_to_run]
        arxiv_dataset.data.edge_index = qq
        print("dataset shape after reduction:")
        
        print(arxiv_dataset.data.x.shape)
        print(arxiv_dataset.data.y.shape)
        print(arxiv_dataset.data.edge_index.shape)

        train_indices = torch.from_numpy(np.array([i for i in range(splits["train"].shape[0])])).type_as(splits["train"])
        valid_indices = torch.from_numpy(np.array([i + train_indices.shape[0] for i in range(splits["valid"].shape[0])])).type_as(splits["valid"])

        splits["train"] = train_indices
        splits["valid"] = valid_indices
        return
    return