import torch
from torch.nn import Linear as Lin, ReLU
from torch.nn import functional as F
from se3_transformer_pytorch import SE3Transformer

class EquivariantGNN_PointNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_node_types):
        super(EquivariantGNN_PointNet, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_node_types, embedding_dim=num_node_features)
        self.transformer1 = SE3Transformer(num_node_features, hidden_dim)
        self.transformer2 = SE3Transformer(hidden_dim, hidden_dim)
        self.lin1 = Lin(hidden_dim, hidden_dim)
        self.lin2 = Lin(hidden_dim, output_dim)
        self.relu = ReLU()

    def forward(self, data):
        x = torch.cat([data.pos, self.embedding(data.z)], dim=-1)  # use embedding for atomic numbers
        x = self.transformer1(x)
        x = self.relu(x)
        x = self.transformer2(x)
        x = self.relu(x)

        # Apply PointNet layers
        x = self.relu(self.lin1(x))
        x = self.lin2(x)

        # Use max pooling for readout to capture the most relevant features
        x, _ = torch.max(x, dim=0)  # max pooling

        return x
