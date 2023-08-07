from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import to_dense_adj
import pickle
import os
from torch.nn.functional import mse_loss
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.loader import DataLoader
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from sklearn.metrics import r2_score
import torch.nn.functional as F

dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'datasetQM9.pkl')

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)


max_num_nodes = 0
dataset_tg = []
for atom_features, atom_positions, _, bond_features, dipole, _ in data:
    # Bond start and end indices
    bond_indices = np.array([(bond[0], bond[1]) for bond in bond_features]).T
    edge_index = torch.tensor(bond_indices, dtype=torch.long)
    
    # Bond types as edge attributes
    bond_types = [bond[2] for bond in bond_features]
    edge_attr = torch.tensor(bond_types, dtype=torch.float).unsqueeze(-1)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    num_nodes = x.shape[0]  # Number of nodes in the current graph
    max_num_nodes = max(max_num_nodes, num_nodes)  # Update if greater than previous max
    pos = torch.tensor(np.array(atom_positions), dtype=torch.float)
    y = torch.tensor(dipole, dtype=torch.float)
    
    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
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


class DipolePredictorE3NN(nn.Module):
    def __init__(self):
        super(DipolePredictorE3NN, self).__init__()

        irreps_in_feats = Irreps("5x0e")  
        irreps_out = Irreps("1x1e")
        irreps_in_coors = Irreps("1x1e")
        
        # Equivariant tensor product
        self.tp = FullyConnectedTensorProduct(irreps_in1=irreps_in_feats, irreps_in2=irreps_in_coors, irreps_out=irreps_out)

        # Fully connected layers for post-processing
        self.fc1 = nn.Linear(irreps_out.dim, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, feats, coors):
        feats_out = self.tp(feats, coors)

        # Mean pooling across nodes, preserving batch and feature dims
        graph_embedding = feats_out.mean(dim=1)

        # Pass through fully connected layers
        graph_embedding = F.relu(self.fc1(graph_embedding))
        output = self.fc2(graph_embedding)

        return output


net = DipolePredictorE3NN()
net.to(device)
for name, value in vars(net).items():
    print(f'{name}: {value}')


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_function_l1 = nn.L1Loss()
best_val_loss = float('inf')
for epoch in range(1000):
    net.train()
    total_loss_l1 = 0
    total_loss_mse = 0
    total_r2 = 0
    for batch in train_loader:
        batch = batch.to(device)
        # Reshape the features and coordinates based on the batch vector
        feats, _ = to_dense_batch(batch.x, batch.batch) # Shape: (batch_size, num_nodes, num_features)
        coors, _ = to_dense_batch(batch.pos, batch.batch) # Shape: (batch_size, num_nodes, 3)

        target = batch.y.view(-1, 3) # Shape: (batch_size, 3)

#        batch_sizes = [torch.sum(batch.batch == i) for i in range(batch.batch.max() + 1)]
        #adj_mat = to_dense_adj(batch.edge_index, batch = batch.batch)
        feats_out = net(feats, coors)
        
        # Compute Loss
        #loss = loss_function(feats_out, target)
        # Compute Losses
        loss_l1 = loss_function_l1(feats_out, target)
        loss_mse = mse_loss(feats_out, target)
        # Backward propagation
        loss_l1.backward()

        # Update parameters
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()
        total_loss_l1 += loss_l1.item()
        total_loss_mse += loss_mse.item()
        r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
        total_r2 += r2

    avg_loss_l1 = total_loss_l1 / len(train_loader)
    avg_r2 = total_r2 / len(train_loader)
    avg_loss_mse = total_loss_mse / len(train_loader)
    print(f'Epoch {epoch}, L1 Loss: {avg_loss_l1}, MSE Loss: {avg_loss_mse}, R2 Score: {avg_r2}')
    
    
    # Validation
    net.eval()
    with torch.no_grad():
        val_loss_l1 = 0
        val_loss_mse = 0
        val_r2 = 0
        for batch in val_loader:
            batch = batch.to(device)
            feats, _ = to_dense_batch(batch.x, batch.batch)
            coors, _ = to_dense_batch(batch.pos, batch.batch)
            target = batch.y.view(-1, 3)
            feats_out = net(feats, coors)

            # Compute Loss
            loss_l1 = loss_function_l1(feats_out, target)
            loss_mse = mse_loss(feats_out, target)
            val_loss_l1 += loss_l1.item()
            val_loss_mse += loss_mse.item()
            # Compute R2 score
            r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
            val_r2 += r2
        avg_val_loss_l1 = val_loss_l1 / len(val_loader)
        avg_val_loss_mse = val_loss_mse / len(val_loader)
        avg_val_r2 = val_r2 / len(val_loader)
        print(f'Validation L1 Loss: {avg_val_loss_l1}, Validation MSE Loss: {avg_val_loss_mse}, Validation R2 Score: {avg_r2}')


        # Save the model if it has the best validation loss so far
        if avg_val_loss_l1 < best_val_loss:
            best_val_loss = avg_val_loss_l1
            torch.save(net.state_dict(), 'best_model_e3nn_dipole_mae.pth')


# Testing
net.load_state_dict(torch.load('best_model_e3nn_dipole_mae.pth'))
net.eval()
with torch.no_grad():
    test_loss_l1 = 0
    test_loss_mse = 0
    test_r2 = 0
    for batch in test_loader:
        batch = batch.to(device)
        feats, _ = to_dense_batch(batch.x, batch.batch)
        coors, _ = to_dense_batch(batch.pos, batch.batch)
        target = batch.y.view(-1, 3)
        feats_out = net(feats, coors)

        # Compute Loss
        loss_l1 = loss_function_l1(feats_out, target)
        test_loss_l1 += loss_l1.item()
        loss_mse = mse_loss(feats_out, target)
        test_loss_mse += loss_mse.item()

        # Compute R2 score
        r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
        test_r2 += r2

    avg_test_loss_l1 = test_loss_l1 / len(test_loader)
    avg_test_r2 = test_r2 / len(test_loader)
    avg_test_loss_mse = test_loss_mse / len(val_loader)
    print(f'Test Loss L1: {avg_test_loss_l1}, MSE Loss: {avg_test_loss_mse}, Test R2 Score: {avg_test_r2}')

# You can load the saved model using `net.load_state_dict(torch.load('best_model.pth'))`
# Then, you can pass your input features to the model using `net(feats, coors, adj_mat=adj_mat)`
# to get the predictions for the dipole moment.        