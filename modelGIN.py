import os
import torch
from torch.nn import functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, DataLoader, Batch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import numpy as np
from QM93D_MM import QM93D
from sklearn.metrics import mean_absolute_error, r2_score


class MyDataLoader(DataLoader):
    def collate(self, data_list):
        batch = super().collate(data_list)

        # Handle extra attributes here
        batch.dipole = torch.stack([data.dipole.view(-1, output_dim_dipoles) for data in data_list])
        batch.quadrupole = torch.stack([data.quadrupole.view(-1, output_dim_quadrupoles) for data in data_list])

        return batch

num_node_features = 8
num_node_types = 5  # number of unique atomic numbers - len(ATOMIC_WEIGHTS)
hidden_dim = 64
output_dim_dipoles = 3  # for dipoles
output_dim_quadrupoles = 9  # for quadrupoles

# Load the dataset
dataset = QM93D(root='data')

# Define the mapping from atomic numbers to indices
atomic_number_to_index = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}

# Split data into train, validation and test
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_data, valid_data, test_data = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

# Create data loaders
train_loader = MyDataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = MyDataLoader(valid_data, batch_size=32)
test_loader = MyDataLoader(test_data, batch_size=32)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GINNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim):
        super(GINNet, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_node_features, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.fc1 = torch.nn.Linear(hidden_dim, output_dim)
        
        # Adding an embedding layer for atom types
        self.embedding = torch.nn.Embedding(len(atomic_number_to_index), num_node_features - 3)  # -3 as we have 3D coordinates already

    def forward(self, data):
        z_indices = torch.tensor([atomic_number_to_index[atomic_number] for atomic_number in data.z.cpu().numpy()]).to(device)
        x = torch.cat([data.pos, self.embedding(z_indices)], dim=-1)  # Combine position and atom type information

        edge_index = data.edge_index
        
        print(f"x shape: {x.shape}, edge_index shape: {edge_index.shape}")
        print(f"x type: {type(x)}, edge_index type: {type(edge_index)}")

        x = F.relu(self.conv1(x, edge_index))
        print("Shape of x before pooling: ", x.shape)
        print("Shape of batch tensor: ", data.batch.shape)
        x = global_add_pool(x, data.batch)
        print("Shape of x after pooling: ", x.shape)

        return self.fc1(x)



# Define models
dipole_model = GINNet(num_node_features, hidden_dim, output_dim_dipoles).to(device)  # 3 for dipole
quadrupole_model = GINNet(num_node_features, hidden_dim, output_dim_quadrupoles).to(device)  # 6 for quadrupole

# Define loss
criterion = torch.nn.MSELoss()

# Define optimizer
optimizer = torch.optim.Adam(list(dipole_model.parameters()) + list(quadrupole_model.parameters()), lr=0.01)

# Training loop
best_valid_loss = float('inf')
patience = 10
epochs_no_improve = 0


num_epochs = 1000

for epoch in range(num_epochs):
    train_loss = 0
    valid_loss = 0

    # Training
    dipole_model.train()
    quadrupole_model.train()
    for batch in train_loader:
        # Print shapes of dipole and quadrupole tensors
        #print("Shape of batch.dipole before reshape: ", batch.dipole.shape)
        #print("Shape of batch.quadrupole before reshape: ", batch.quadrupole.shape)
        # Reshape batch.dipole and batch.quadrupole
        batch.dipole = batch.dipole.view(-1, output_dim_dipoles).to(device)
        batch.quadrupole = batch.quadrupole.view(-1, output_dim_quadrupoles).to(device)
        #print("Shape of batch.dipole after reshape: ", batch.dipole.shape)
        #print("Shape of batch.quadrupole after reshape: ", batch.quadrupole.shape)
        # Forward pass
        batch = batch.to(device)
        dipole_pred = dipole_model(batch)
        quadrupole_pred = quadrupole_model(batch)

        #print("Shape of predicted dipole: ", dipole_pred.shape)
        #print("Shape of ground truth dipole: ", batch.dipole.shape)

        # Compute loss
        dipole_loss = criterion(dipole_pred, batch.dipole)
        quadrupole_loss = criterion(quadrupole_pred, batch.quadrupole)

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss = dipole_loss + quadrupole_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += total_loss.item()

    # Validation
    dipole_model.eval()
    quadrupole_model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            batch.dipole = batch.dipole.view(-1, output_dim_dipoles).to(device)
            batch.quadrupole = batch.quadrupole.view(-1, output_dim_quadrupoles).to(device)
            
            batch = batch.to(device)
            dipole_pred = dipole_model(batch)
            quadrupole_pred = quadrupole_model(batch)
            
            # Compute loss
            dipole_loss = criterion(dipole_pred, batch.dipole)
            quadrupole_loss = criterion(quadrupole_pred, batch.quadrupole)
            
            total_loss = dipole_loss + quadrupole_loss
            valid_loss += total_loss.item()

    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    # Early stopping
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        epochs_no_improve = 0
        # Save the model
        torch.save({'dipole_model_state_dict': dipole_model.state_dict(),
                    'quadrupole_model_state_dict': quadrupole_model.state_dict()}, 'best_model_gin.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print('Early stopping!')
            break

# Load the best model
checkpoint = torch.load('best_model.pth')
dipole_model.load_state_dict(checkpoint['dipole_model_state_dict'])
quadrupole_model.load_state_dict(checkpoint['quadrupole_model_state_dict'])

# Test the model
dipole_model.eval()
quadrupole_model.eval()

actuals_dipole, preds_dipole = [], []
actuals_quadrupole, preds_quadrupole = [], []

with torch.no_grad():
    for batch in test_loader:
        batch.dipole = batch.dipole.view(-1, output_dim_dipoles).to(device)
        batch.quadrupole = batch.quadrupole.view(-1, output_dim_quadrupoles).to(device)
        
        batch = batch.to(device)
        dipole_pred = dipole_model(batch)
        quadrupole_pred = quadrupole_model(batch)

        actuals_dipole.append(batch.dipole.cpu().numpy())
        preds_dipole.append(dipole_pred.cpu().numpy())
        actuals_quadrupole.append(batch.quadrupole.cpu().numpy())
        preds_quadrupole.append(quadrupole_pred.cpu().numpy())

# Flatten the lists of numpy arrays
actuals_dipole = np.concatenate(actuals_dipole)
preds_dipole = np.concatenate(preds_dipole)
actuals_quadrupole = np.concatenate(actuals_quadrupole)
preds_quadrupole = np.concatenate(preds_quadrupole)

# Calculate metrics for dipoles
mae_dipole = [mean_absolute_error(actuals_dipole[:, i], preds_dipole[:, i]) for i in range(output_dim_dipoles)]
r2_dipole = [r2_score(actuals_dipole[:, i], preds_dipole[:, i]) for i in range(output_dim_dipoles)]
# Calculate metrics for quadrupoles
mae_quadrupole = [mean_absolute_error(actuals_quadrupole[:, i], preds_quadrupole[:, i]) for i in range(output_dim_quadrupoles)]
r2_quadrupole = [r2_score(actuals_quadrupole[:, i], preds_quadrupole[:, i]) for i in range(output_dim_quadrupoles)]

print(f'Dipole Model: MAE = {mae_dipole}, R^2 = {r2_dipole}')
print(f'Quadrupole Model: MAE = {mae_quadrupole}, R^2 = {r2_quadrupole}')
