import pickle
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.data import DataLoader
from egnn_pytorch import EGNN_Sparse
from sklearn.metrics import r2_score

from torch import nn
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import BatchNorm

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


class CustomModel(nn.Module):
    def __init__(self, feats_dim, pos_dim=3, edge_attr_dim=0, m_dim=16, fourier_features=0):
        super(CustomModel, self).__init__()
        
        # Initialize the EGNN_Sparse layer
        self.gnn = EGNN_Sparse(
            feats_dim=feats_dim,
            pos_dim=pos_dim,
            edge_attr_dim=edge_attr_dim,
            m_dim=m_dim,
            fourier_features=fourier_features
        )
        
        # Add additional layers as needed
        self.bn = BatchNorm(feats_dim)
        self.fc = nn.Linear(feats_dim, 3) # Output shape adjusted to 3 for dipole


    def forward(self, x, edge_index, pos, batch, edge_attr=None):
        # Concatenate pos and x to match the expected input of EGNN_Sparse
        x_combined = torch.cat([pos, x], dim=-1)

        # Call EGNN_Sparse with the concatenated tensor
        x_out = self.gnn(x_combined, edge_index, edge_attr=edge_attr, batch=batch)

        # Split the output back into coordinates and features if needed
        coors_out, hidden_out = x_out[:, :self.pos_dim], x_out[:, self.pos_dim:]
        
        # Apply BatchNorm to the features
        hidden_out = self.bn(hidden_out, batch)
        out = self.fc(hidden_out) # Use fully connected layer
        return out
    

    # Instantiate the custom model
#feats_dim = x.shape[1]

#sample_batch = next(iter(train_loader))
#feats_dim = sample_batch.x.shape[1] - 3 # -3 to exclude pos dimensions
model = CustomModel(feats_dim=1)

# Define a loss function, optimizer, etc.
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(1000):
    for batch in train_loader:
        optimizer.zero_grad()
        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.pos, batch.batch)
        # Compute loss
        loss = loss_func(pred, batch.y)
        # Backward pass
        loss.backward()
        optimizer.step()



# Validation and testing loop
def evaluate(loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.x, batch.edge_index, batch.pos, batch.batch)
            predictions.append(pred)
            targets.append(batch.y)
    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    return r2_score(targets.numpy(), predictions.numpy())

# Training loop
for epoch in range(1000):
    for batch in train_loader:
        optimizer.zero_grad()
        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.pos, batch.batch)
        # Compute loss
        loss = loss_func(pred, batch.y)
        # Backward pass
        loss.backward()
        optimizer.step()   

    # Validate and print metrics
    train_r2 = evaluate(train_loader)
    val_r2 = evaluate(val_loader)
    print(f"Epoch {epoch}: Train R2: {train_r2}, Validation R2: {val_r2}")

# Test and print metrics
test_r2 = evaluate(test_loader)
print(f"Test R2: {test_r2}")

# Save the model
torch.save(model.state_dict(), 'model.pth')