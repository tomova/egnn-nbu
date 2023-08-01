import pickle
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from egnn_pytorch import EGNN_Network
from sklearn.metrics import r2_score
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'dataset.pkl')

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)
    

# Create a PyTorch Geometric dataset
dataset_tg = []
for atom_features, adjacency_matrix, atom_positions, dipole, quadrupole in data:
    #edge_index = torch.tensor(np.where(adjacency_matrix == 1), dtype=torch.long)
    edge_index = torch.tensor(np.stack(np.where(adjacency_matrix == 1)), dtype=torch.long)

    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    pos = torch.tensor(atom_positions, dtype=torch.float)
    y = {"dipole": torch.tensor(dipole, dtype=torch.float),
         "quadrupole": torch.tensor(quadrupole, dtype=torch.float)}
    graph_data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    dataset_tg.append(graph_data)

# Define the split sizes
train_size = 110000
val_size = 10000
test_size = len(dataset_tg) - train_size - val_size

# Split the data using train_test_split with specific random seed
train_data, temp_data = train_test_split(dataset_tg[:train_size + val_size], train_size=train_size, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=test_size, random_state=42)

# Print the size of each split
print("Train size:", len(train_data))
print("Validation size:", len(val_data))
print("Test size:", len(test_data))

# Define DataLoader objects
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)


# Custom Network Class
class CustomEGNN_Network(EGNN_Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dipole_head = nn.Linear(kwargs['dim'], 3) # Output for dipole
        self.quadrupole_head = nn.Linear(kwargs['dim'], 9) # Output for quadrupole

    def forward(self, data):
        feats, coors = super().forward(feats=data.x, coors=data.pos, adj_mat=data.edge_index)
        dipole = self.dipole_head(feats)
        quadrupole = self.quadrupole_head(feats)
        return dipole, quadrupole

# Network Initialization
model = CustomEGNN_Network(
    depth=4,
    dim=64,
    num_positions=3
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(1000):
    model.train()
    total_loss = 0
    total_r2_dipole = 0
    total_r2_quadrupole = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        dipole_pred, quadrupole_pred = model(batch)
        loss_dipole = criterion(dipole_pred, batch.y["dipole"])
        loss_quadrupole = criterion(quadrupole_pred, batch.y["quadrupole"])
        loss = loss_dipole + loss_quadrupole
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_r2_dipole += r2_score(dipole_pred.detach().cpu().numpy(), batch.y["dipole"].cpu().numpy())
        total_r2_quadrupole += r2_score(quadrupole_pred.detach().cpu().numpy(), batch.y["quadrupole"].cpu().numpy())

        #total_r2_dipole += r2_score(dipole_pred.detach().numpy(), batch.y["dipole"].numpy())
        #total_r2_quadrupole += r2_score(quadrupole_pred.detach().numpy(), batch.y["quadrupole"].numpy())

    avg_loss = total_loss / len(train_loader)
    avg_r2_dipole = total_r2_dipole / len(train_loader)
    avg_r2_quadrupole = total_r2_quadrupole / len(train_loader)
    
    print(f'Epoch {epoch}, Loss: {avg_loss}, R2 Dipole: {avg_r2_dipole}, R2 Quadrupole: {avg_r2_quadrupole}')


# Evaluation on the Test Set
model.eval()
with torch.no_grad():
    mse_dipole, mse_quadrupole, r2_dipole, r2_quadrupole = 0, 0, 0, 0
    for batch in test_loader:
        batch = batch.to(device)
        dipole_pred, quadrupole_pred = model(batch)
        mse_dipole += criterion(dipole_pred, batch.y["dipole"]).item()
        mse_quadrupole += criterion(quadrupole_pred, batch.y["quadrupole"]).item()
        r2_dipole += r2_score(dipole_pred.numpy(), batch.y["dipole"].cpu().numpy())
        r2_quadrupole += r2_score(quadrupole_pred.numpy(), batch.y["quadrupole"].cpu().numpy())

    mse_dipole /= len(test_loader)
    mse_quadrupole /= len(test_loader)
    r2_dipole /= len(test_loader)
    r2_quadrupole /= len(test_loader)

    print(f"MSE Dipole: {mse_dipole}, MSE Quadrupole: {mse_quadrupole}")
    print(f"R2 Dipole: {r2_dipole}, R2 Quadrupole: {r2_quadrupole}")

# Saving the Model
torch.save(model.state_dict(), 'model_egnn.pth')

# To load the model in the future:
# model = CustomEGNN_Network(
#     depth=4,
#     dim=64,
#     num_positions=3
# )
# model.load_state_dict(torch.load('model_egnn.pth'))
# model.eval()