import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_add_pool
from torch.nn import functional as F

class EGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, node_dim):
        super(EGNNConv, self).__init__(aggr='add')
        self.lin_node = torch.nn.Linear(in_channels, out_channels)
        self.lin_edge_attr = torch.nn.Linear(edge_dim, out_channels)
        self.lin_edge_attr2 = torch.nn.Linear(edge_dim, node_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        edge_attr = self.lin_edge_attr(edge_attr)
        return self.propagate(edge_index, x=self.lin_node(x), edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class EquivariantGNN(torch.nn.Module):
    def __init__(self, hidden_channels, edge_dim, node_dim, num_node_types):
        super(EquivariantGNN, self).__init__()
        self.egnn1 = EGNNConv(num_node_types + 3, hidden_channels, edge_dim, node_dim) # 3 for 3D position
        self.egnn2 = EGNNConv(hidden_channels, hidden_channels, edge_dim, node_dim)
        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.num_node_types = num_node_types

    def forward(self, pos, z, edge_index, edge_attr, batch):
        x = torch.cat((pos, F.one_hot(z, num_classes=self.num_node_types).float()), dim=1) # Concatenate position and one-hot encoded atomic number
        x = F.relu(self.egnn1(x, edge_index, edge_attr))
        x = F.relu(self.egnn2(x, edge_index, edge_attr))
        x = global_add_pool(x, batch)
        return self.lin(x)
