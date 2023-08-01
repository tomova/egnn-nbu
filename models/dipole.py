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

dataset_tg = []
for atom_features, adjacency_matrix, atom_positions, dipole, _ in data:
    edge_index = torch.tensor(np.stack(np.where(adjacency_matrix == 1)), dtype=torch.long)
    x = torch.tensor(atom_features, dtype=torch.long)  # Use Long dtype
    #x = torch.tensor(atom_features, dtype=torch.long).unsqueeze(1)
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

net = EGNN_Network(
    num_tokens=10, # or other suitable value
    dim=32,
    depth=3
)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

for epoch in range(1000):
    total_loss = 0
    #total_r2_dipole = 0
    for batch in train_loader:
        # Extract batch data
        feats = batch.x
        coors = batch.pos
        adj_mat = batch.edge_index
        target = batch.y

        # Forward propagation
        feats_out, coors_out = net(feats, coors, adj_mat=adj_mat)

        # Compute Loss
        loss = loss_function(feats_out, target)

        # Backward propagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()
        total_loss += loss.item()
        #total_r2_dipole += r2_score(dipole_pred.detach().cpu().numpy(), batch.y.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    #avg_r2_dipole = total_r2_dipole / len(train_loader)
    print(f'Epoch {epoch}, Loss: {avg_loss}')
    #print(f'Epoch {epoch}, Loss: {avg_loss}, R2 Dipole: {avg_r2_dipole}')