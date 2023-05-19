import torch
from torch.nn import Linear as Lin
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv

class EquivariantGNN_GAT(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_node_types):
        super(EquivariantGNN_GAT, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_node_types, embedding_dim=num_node_features)
        self.gat1 = GATConv(num_node_features, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.lin = Lin(hidden_dim, output_dim)

    def forward(self, data):
        x = torch.cat([data.pos, self.embedding(data.z)], dim=-1)  # use embedding for atomic numbers
        x = F.elu(self.gat1(x, data.edge_index))
        x = F.elu(self.gat2(x, data.edge_index))
        x = self.lin(x)

        # Use mean for readout
        x = global_mean_pool(x, data.batch)

        return x
