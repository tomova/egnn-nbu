import torch
from torch.nn import Linear as Lin, ReLU
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool
from se3_transformer_pytorch import SE3Transformer

# Define the mapping from atomic numbers to indices
atomic_number_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EquivariantGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_node_types):
        super(EquivariantGNN, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=num_node_types, embedding_dim=num_node_features)

        self.transformer = SE3Transformer(
            dim=num_node_features + 3,  # 3 for the 3D position  
            heads=8,
            dim_head=16, 
            num_degrees=4
        )

        self.lin = Lin(hidden_dim, output_dim)

    def forward(self, data):
        # Map atomic numbers to indices
        z_indices = torch.tensor([atomic_number_to_index[atomic_number] for atomic_number in data.z.cpu().numpy()]).to(device)

        x = torch.cat([data.pos, self.embedding(z_indices)], dim=-1)  # Combine position and atom type information
        x = x.reshape(1, *x.shape)
        edges = data.edge_index.reshape(1, *data.edge_index.shape)
        edge_attr = data.edge_attr.reshape(1, *data.edge_attr.shape)
        # x = x.unsqueeze(0)
        # edges = data.edge_index.unsqueeze(0)
        # edge_attr = data.edge_attr.unsqueeze(0)

        x = self.transformer(x, edges, edge_attr)
        x = x.tensor
        x = F.relu(x)
        x = self.lin(x)
        x = global_mean_pool(x, data.batch)
        return x