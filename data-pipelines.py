from torch_geometric.datasets import Planetoid, ZINC, QM9
from torch_geometric.transforms import NormalizeFeatures


def get_and_clean_CORA():
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())   ## This should be fine! 

    return


def get_and_clean_ZINC():
    zinc_dataset = ZINC(root='data/ZINC', transform=NormalizeFeatures()) 

    return

def get_and_clean_QM9():
    QM9_dataset = QM9(root='data/QM9', name='QM9', transform=NormalizeFeatures()) 

    return


if __name__ == "__main__":
    a=2