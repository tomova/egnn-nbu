from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import to_dense_adj
import pickle
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from se3_transformer_pytorch import SE3Transformer

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'dataset.pkl')

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)


max_num_nodes = 0
dataset_tg = []
for atom_features, adjacency_matrix, atom_positions, dipole, _ in data:
    edge_index = torch.tensor(np.stack(np.where(adjacency_matrix == 1)), dtype=torch.long)

    x = torch.tensor(atom_features, dtype=torch.long).unsqueeze(-1) 

    num_nodes = x.shape[0]  # Number of nodes in the current graph
    max_num_nodes = max(max_num_nodes, num_nodes)  # Update if greater than previous max
    pos = torch.tensor(np.array(atom_positions), dtype=torch.float)
    y = torch.tensor(dipole, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    dataset_tg.append(graph_data)

print("Maximum number of atoms:", max_num_nodes)
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GlobalSumPooling(nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


class DipolePredictorSE3(nn.Module):
    def __init__(self, output_dim=3, num_layers=3, hidden_dim=32):
        super(DipolePredictorSE3, self).__init__()
        
        self.se3_transformer = SE3Transformer(
            num_layers = num_layers, # Specify the number of layers
            input_dim = 3, # Assume input features are 3D coordinates
            hidden_dim = hidden_dim, # Hidden feature dimension
            # Include any other specific parameters required for your use case
        )

        # A fully connected layer to produce the final output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, feats, coors, adj_mat):
        # Depending on the exact API, you might need to provide more or different inputs
        x_transformed = self.se3_transformer(feats, coors, adj_mat)
        
        # Take the mean across the nodes, preserving batch and feature dims
        x_pooled = x_transformed.mean(dim=1)
        
        # Pass through the fully connected layers
        output = self.fc(x_pooled)

        return output




net = DipolePredictorSE3()
net.to(device)
for name, value in vars(net).items():
    print(f'{name}: {value}')


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.L1Loss()
best_val_loss = float('inf')
for epoch in range(1000):
    net.train()
    total_loss = 0
    total_r2 = 0
    for batch in train_loader:
        batch = batch.to(device)
        # Reshape the features and coordinates based on the batch vector
        feats, _ = to_dense_batch(batch.x, batch.batch) # Shape: (batch_size, num_nodes, num_features)
        coors, _ = to_dense_batch(batch.pos, batch.batch) # Shape: (batch_size, num_nodes, 3)

        target = batch.y.view(-1, 3) # Shape: (batch_size, 3)

        batch_sizes = [torch.sum(batch.batch == i) for i in range(batch.batch.max() + 1)]
        adj_mat = to_dense_adj(batch.edge_index, batch = batch.batch)
        feats_out = net(feats, coors, adj_mat=adj_mat)
        
        # Compute Loss
        loss = loss_function(feats_out, target)

        # Backward propagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()
        total_loss += loss.item()
        r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
        total_r2 += r2

    avg_loss = total_loss / len(train_loader)
    avg_r2 = total_r2 / len(train_loader)
    print(f'Epoch {epoch}, Loss: {avg_loss}, R2 Score: {avg_r2}')

    # Validation
    net.eval()
    with torch.no_grad():
        val_loss = 0
        val_r2 = 0
        for batch in val_loader:
            batch = batch.to(device)
            feats, _ = to_dense_batch(batch.x, batch.batch)
            coors, _ = to_dense_batch(batch.pos, batch.batch)
            target = batch.y.view(-1, 3)
            adj_mat = to_dense_adj(batch.edge_index, batch=batch.batch)
            feats_out = net(feats, coors, adj_mat=adj_mat)

            # Compute Loss
            loss = loss_function(feats_out, target)
            val_loss += loss.item()

            # Compute R2 score
            r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
            val_r2 += r2
        avg_val_loss = val_loss / len(val_loader)
        avg_val_r2 = val_r2 / len(val_loader)
        print(f'Validation Loss: {avg_val_loss}, Validation R2 Score: {avg_val_r2}')

        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), 'best_model_egnn_dipole_mae.pth')


# Testing
net.load_state_dict(torch.load('best_model_egnn_dipole_mae.pth'))
net.eval()
with torch.no_grad():
    test_loss = 0
    test_r2 = 0
    for batch in test_loader:
        batch = batch.to(device)
        feats, _ = to_dense_batch(batch.x, batch.batch)
        coors, _ = to_dense_batch(batch.pos, batch.batch)
        target = batch.y.view(-1, 3)
        adj_mat = to_dense_adj(batch.edge_index, batch=batch.batch)
        feats_out = net(feats, coors, adj_mat=adj_mat)

        # Compute Loss
        loss = loss_function(feats_out, target)
        test_loss += loss.item()

        # Compute R2 score
        r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
        test_r2 += r2

    avg_test_loss = test_loss / len(test_loader)
    avg_test_r2 = test_r2 / len(test_loader)
    print(f'Test Loss: {avg_test_loss}, Test R2 Score: {avg_test_r2}')

# You can load the saved model using `net.load_state_dict(torch.load('best_model.pth'))`
# Then, you can pass your input features to the model using `net(feats, coors, adj_mat=adj_mat)`
# to get the predictions for the dipole moment.        