import torch
from torch.nn import Linear as Lin, ReLU
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from se3_transformer_pytorch import SE3Transformer

class EquivariantGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_node_types):
        super(EquivariantGNN, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_node_types, embedding_dim=num_node_features)
        self.transformer1 = SE3Transformer(num_node_features, hidden_dim)
        self.transformer2 = SE3Transformer(hidden_dim, hidden_dim)
        self.lin = Lin(hidden_dim, output_dim)

    def forward(self, data):
        x = torch.cat([data.pos, self.embedding(data.z)], dim=-1)  # use embedding for atomic numbers
        x = self.transformer1(x)
        x = x.relu()
        x = self.transformer2(x)
        x = x.relu()
        x = self.lin(x)

        # Use mean for readout
        x = global_mean_pool(x, data.batch)

        return x
