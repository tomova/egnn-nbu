import pickle
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.data import DataLoader
from egnn_pytorch import EGNN_Sparse_Network
from sklearn.metrics import r2_score

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'dataset.pkl')

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)

dataset_tg = []
for atom_features, adjacency_matrix, atom_positions, dipole, _ in data:
    edge_index = torch.tensor(np.stack(np.where(adjacency_matrix == 1)), dtype=torch.long)
    #x = torch.tensor(atom_features, dtype=torch.long)  # Use Long dtype
    x = torch.tensor(atom_features, dtype=torch.long).unsqueeze(1)
    pos = torch.tensor(np.array(atom_positions), dtype=torch.float)
    y = torch.tensor(dipole, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    dataset_tg.append(graph_data)

# Define the split sizes
total_samples = len(dataset_tg)
train_size = 110000
val_size = 10000
test_size = total_samples - train_size - val_size

# Split the data using train_test_split with specific random seed
train_data, temp_data = train_test_split(dataset_tg, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)

# Print the size of each split
print("Train size:", len(train_data))
print("Validation size:", len(val_data))
print("Test size:", len(test_data))


# Define DataLoader objects
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

feats_dim = 1
pos_dim = 3
edge_attr_dim = 0 # Assuming you don't have edge attributes

model = EGNN_Sparse_Network(
    n_layers=3, 
    feats_dim=feats_dim, 
    pos_dim=pos_dim,
    edge_attr_dim=edge_attr_dim,
    m_dim=16,
    fourier_features=0, 
    soft_edge=0,
    update_coors=True,
    update_feats=True,
    norm_feats=True,
    norm_coors=False,
    dropout=0.0,
    aggr="add"
)

class DipolePredictionModel(nn.Module):
    def __init__(self, gnn, output_dim):
        super().__init__()
        self.gnn = gnn
        self.regression_head = nn.Sequential(
            nn.Linear(feats_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.gnn(x, edge_index, batch, edge_attr)
        x = torch.mean(x, dim=0) # Average the node features to get graph-level representation
        x = self.regression_head(x)
        return x.squeeze(-1) # Output shape: (batch_size,)

model = DipolePredictionModel(model, output_dim=1)


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
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        print("Data.pos shape:", batch.pos.shape)
        dipole_pred = model(batch)
        loss_dipole = criterion(dipole_pred, batch.y)
        loss_dipole.backward()
        optimizer.step()

        total_loss += loss_dipole.item()
        total_r2_dipole += r2_score(dipole_pred.detach().cpu().numpy(), batch.y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    avg_r2_dipole = total_r2_dipole / len(train_loader)
    
    print(f'Epoch {epoch}, Loss: {avg_loss}, R2 Dipole: {avg_r2_dipole}')


# Evaluation on the Test Set
model.eval()
with torch.no_grad():
    mse_dipole, r2_dipole = 0, 0
    for batch in test_loader:
        batch = batch.to(device)
        dipole_pred = model(batch)
        mse_dipole += criterion(dipole_pred, batch.y).item()
        r2_dipole += r2_score(dipole_pred.cpu().numpy(), batch.y.cpu().numpy())

    mse_dipole /= len(test_loader)
    r2_dipole /= len(test_loader)

    print(f"MSE Dipole: {mse_dipole}")
    print(f"R2 Dipole: {r2_dipole}")

# Saving the Model
torch.save(model.state_dict(), 'model_egnn_dipole.pth')
