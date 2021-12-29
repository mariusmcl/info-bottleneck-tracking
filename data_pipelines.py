from numpy.core.fromnumeric import prod
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def get_data_and_indices(dataset_name):
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
        
    else:
        print(f"Dataset {dataset_name} not found")
