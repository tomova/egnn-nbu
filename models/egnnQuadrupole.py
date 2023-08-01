import pickle
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.data import DataLoader
from egnn_pytorch import EGNN_Network
from sklearn.metrics import r2_score

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'dataset.pkl')

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

# Create a PyTorch Geometric dataset
dataset_tg = []
for atom_features, adjacency_matrix, atom_positions, _, quadrupole in data:
    edge_index = torch.tensor(np.stack(np.where(adjacency_matrix == 1)), dtype=torch.long)
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)
    pos = torch.tensor(atom_positions, dtype=torch.float)
    y = torch.tensor(quadrupole, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    dataset_tg.append(graph_data)

# Define the split sizes
train_size = 110000
val_size = 10000
test_size = len(dataset_tg) - train_size - val_size

# Split the data using train_test_split with specific random seed
train_data, temp_data = train_test_split(dataset_tg[:train_size + val_size], train_size=train_size, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=test_size, random_state=42)

# Define DataLoader objects
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)


# Custom Network Class
class CustomEGNN_Network(EGNN_Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quadrupole_head = nn.Linear(kwargs['dim'], 9) # Output for quadrupole

    def forward(self, data):
        feats, coors = super().forward(feats=data.x, coors=data.pos, adj_mat=data.edge_index)
        quadrupole = self.quadrupole_head(feats)
        return quadrupole

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
    total_r2_quadrupole = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        quadrupole_pred = model(batch)
        loss_quadrupole = criterion(quadrupole_pred, batch.y)
        loss_quadrupole.backward()
        optimizer.step()

        total_loss += loss_quadrupole.item()
        total_r2_quadrupole += r2_score(quadrupole_pred.detach().cpu().numpy(), batch.y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    avg_r2_quadrupole = total_r2_quadrupole / len(train_loader)
    
    print(f'Epoch {epoch}, Loss: {avg_loss}, R2 Quadrupole: {avg_r2_quadrupole}')


# Evaluation on the Test Set
model.eval()
with torch.no_grad():
    mse_quadrupole, r2_quadrupole = 0, 0
    for batch in test_loader:
        batch = batch.to(device)
        quadrupole_pred = model(batch)
        mse_quadrupole += criterion(quadrupole_pred, batch.y).item()
        r2_quadrupole += r2_score(quadrupole_pred.cpu().numpy(), batch.y.cpu().numpy())

    mse_quadrupole /= len(test_loader)
    r2_quadrupole /= len(test_loader)

    print(f"MSE Quadrupole: {mse_quadrupole}")
    print(f"R2 Quadrupole: {r2_quadrupole}")

# Saving the Model
torch.save(model.state_dict(), 'model_quadrupole.pth')
