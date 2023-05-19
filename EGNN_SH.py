import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear as Lin
from torch_geometric.nn import MessagePassing

num_node_types = 5

class SphericalHarmonicsLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_degrees):
        super(SphericalHarmonicsLayer, self).__init__(aggr='add')

        self.num_degrees = num_degrees
        self.lin = Lin(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i):
        return self.lin(x_j - x_i)

    def update(self, aggr_out):
        return F.relu(aggr_out)

class EGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim_dipoles, output_dim_quadrupoles, num_degrees):
        super(EGNN, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=num_node_types, embedding_dim=num_node_features)

        self.sh_layer = SphericalHarmonicsLayer(num_node_features, hidden_dim, num_degrees)
        self.lin_dipoles = Lin(hidden_dim, output_dim_dipoles)
        self.lin_quadrupoles = Lin(hidden_dim, output_dim_quadrupoles)

    def forward(self, data):
        x = torch.cat([data.pos, self.embedding(data.z)], dim=-1)
        x = self.sh_layer(x, data.edge_index)
        x = F.relu(x)
        dipole_pred = self.lin_dipoles(x)
        quadrupole_pred = self.lin_quadrupoles(x)
        return dipole_pred, quadrupole_pred
