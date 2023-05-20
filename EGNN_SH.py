import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear as Lin
from torch_geometric.nn import MessagePassing

num_node_types = 5
# Define the mapping from atomic numbers to indices
atomic_number_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
output_dim_dipoles = 3  # for dipoles
output_dim_quadrupoles = 6  # for dipoles
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
        # Map atomic numbers to indices
        z_indices = torch.tensor([atomic_number_to_index[atomic_number] for atomic_number in data.z.cpu().numpy()]).to(device)

        x = torch.cat([data.pos, self.embedding(z_indices)], dim=-1)  # Combine position and atom type information

        x = self.sh_layer(x, data.edge_index)
        x = F.relu(x)
        dipole_pred = self.lin_dipoles(x.view(-1, self.lin_dipoles.in_features))
        quadrupole_pred = self.lin_quadrupoles(x.view(-1, self.lin_quadrupoles.in_features))
        
        dipole_pred = dipole_pred.view(-1, output_dim_dipoles)
        quadrupole_pred = quadrupole_pred.view(-1, output_dim_quadrupoles)
        
        return dipole_pred, quadrupole_pred
