import torch
from torch.nn import Linear as Lin, ReLU
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from se3_transformer_pytorch import SE3Transformer, SE3T

class EquivariantGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_node_types):
        super(EquivariantGNN, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=num_node_types, embedding_dim=num_node_features)

        # The SE3Transformer requires a list of dimensions for each layer
        self.transformer = SE3Transformer(
            dim=num_node_features,
            heads=4,
            dim_head=16,
            layers=(2,2,2),
            num_degrees=4
        )

        self.lin = Lin(hidden_dim, output_dim)

    def forward(self, data):
        x = torch.cat([data.pos, self.embedding(data.z)], dim=-1)  # use embedding for atomic numbers
        # we need to convert the data to the format expected by the SE3Transformer
        x = SE3T(x, L=data.edge_attr)
        x = self.transformer(x)
        # Convert the SE3T object back into a tensor
        x = x.tensor
        x = F.relu(x)
        x = self.lin(x)
        x = global_mean_pool(x, data.batch)
        return x