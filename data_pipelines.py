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
        REDUCTION_FACTOR = 8 if reduction_factor is None else reduction_factor
        splits["train"] = splits["train"][::REDUCTION_FACTOR]   # Went out of memory (33GB) when comupting full
        splits["valid"] = splits["valid"][::REDUCTION_FACTOR]   
        
        print(splits["valid"].shape)
        print(splits["train"].shape)
        idx_to_run = torch.cat((splits["train"], splits["valid"]), 0)
        print(idx_to_run.shape)
        
        print("dataset shape before reduction:")
        print(arxiv_dataset.data.x.shape)
        print(arxiv_dataset.data.y.shape)
        arxiv_dataset.data.x = arxiv_dataset.data.x[idx_to_run]
        arxiv_dataset.data.y = arxiv_dataset.data.y[idx_to_run]
        print("dataset shape after reduction:")
        
        print(arxiv_dataset.data.x.shape)
        print(arxiv_dataset.data.y.shape)
        
        
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
        
    else:
        print(f"Dataset {dataset_name} not found")
