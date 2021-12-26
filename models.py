import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv




class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_features, activation="relu"):
        def act_func(activation):
            if activation == "relu":
                return nn.ReLU()
            elif activation == "tanh":
                return nn.Tanh()
        super().__init__()
        self.gat_conv1 = GATConv(num_features, hidden_channels)
        self.act1 = act_func(activation=activation) 
        self.gat_conv2 = GATConv(hidden_channels, hidden_channels)
        self.act2 = act_func(activation=activation) 
        self.gat_conv3 = GATConv(hidden_channels, hidden_channels)
        self.act3 = act_func(activation=activation) 
        self.gat_conv4 = GATConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):

        x = self.gat_conv1(x, edge_index)
        x = self.act1(x)

        x = self.gat_conv2(x, edge_index)
        x = self.act2(x)

        x = self.gat_conv3(x, edge_index)
        x = self.act3(x)

        x = self.gat_conv4(x, edge_index)
        return x



class GIN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_features, activation="relu"):
        def act_func(activation):
            if activation == "relu":
                return nn.ReLU()
            elif activation == "tanh":
                return nn.Tanh()
        super().__init__()
        self.gin_conv1 = GINConv(num_features, hidden_channels)
        self.act1 = act_func(activation=activation) 
        self.gin_conv2 = GINConv(hidden_channels, hidden_channels)
        self.act2 = act_func(activation=activation) 
        self.gin_conv3 = GINConv(hidden_channels, hidden_channels)
        self.act3 = act_func(activation=activation) 
        self.gin_conv4 = GINConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):

        x = self.gin_conv1(x, edge_index)
        x = self.act1(x)

        x = self.gin_conv2(x, edge_index)
        x = self.act2(x)

        x = self.gin_conv3(x, edge_index)
        x = self.act3(x)

        x = self.gin_conv4(x, edge_index)
        return x




class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_features, activation="relu"):
        def act_func(activation):
            if activation == "relu":
                return nn.ReLU()
            elif activation == "tanh":
                return nn.Tanh()
        super().__init__()
        self.gcn_conv1 = GCNConv(num_features, hidden_channels)
        self.act1 = act_func(activation=activation) 
        self.gcn_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.act2 = act_func(activation=activation) 
        self.gcn_conv3 = GCNConv(hidden_channels, hidden_channels)
        self.act3 = act_func(activation=activation) 
        self.gcn_conv4 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):

        x = self.gcn_conv1(x, edge_index)
        x = self.act1(x)

        x = self.gcn_conv2(x, edge_index)
        x = self.act2(x)

        x = self.gcn_conv3(x, edge_index)
        x = self.act3(x)

        x = self.gcn_conv4(x, edge_index)
        return x


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes, num_features, activation="relu"):
        def act_func(activation):
            if activation == "relu":
                return nn.ReLU()
            elif activation == "tanh":
                return nn.Tanh()

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_channels)
        self.act1 = act_func(activation=activation) 
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.act2 = act_func(activation=activation) 
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.act3 = act_func(activation=activation) 
        self.fc4 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = self.act3(x)

        x = self.fc4(x)
        return x
