from QM93D_MM import QM93D
from torch_geometric.data import DataLoader
from EGNN import EquivariantGNN
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import torch

num_node_features = 5
num_node_types = 5  # number of unique atomic numbers - len(ATOMIC_WEIGHTS)
hidden_dim = 64
output_dim_dipoles = 3  # for dipoles
output_dim_quadrupoles = 6  # for dipoles


# Load the dataset
dataset = QM93D(root='data')

# Split data into train, validation and test
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_data, valid_data, test_data = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
dipole_model = EquivariantGNN(num_node_features, hidden_dim, output_dim_dipoles, num_node_types).to(device)  # 3 for dipole
quadrupole_model = EquivariantGNN(num_node_features, hidden_dim, output_dim_quadrupoles, num_node_types).to(device)  # 6 for quadrupole

# Define loss
criterion = torch.nn.MSELoss()

# Define optimizer
optimizer = torch.optim.Adam(list(dipole_model.parameters()) + list(quadrupole_model.parameters()), lr=0.01)

# Training loop
best_valid_loss = float('inf')
patience = 10
epochs_no_improve = 0

for epoch in range(100):
    train_loss = 0
    valid_loss = 0

    # Training
    dipole_model.train()
    quadrupole_model.train()
    for batch in train_loader:
        # Forward pass
        batch = batch.to(device)
        dipole_pred = dipole_model(batch)
        quadrupole_pred = quadrupole_model(batch)
        
        # Compute loss
        dipole_loss = criterion(dipole_pred, batch.dipole)
        quadrupole_loss = criterion(quadrupole_pred, batch.quadrupole)

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss = dipole_loss + quadrupole_loss
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()

    # Validation
    dipole_model.eval()
    quadrupole_model.eval()
    with torch.no_grad():
        for batch in valid_loader:
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
                    'quadrupole_model_state_dict': quadrupole_model.state_dict()}, 'best_model.pth')
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
mae_quadrupole = [[mean_absolute_error(actuals_quadrupole[:, i, j], preds_quadrupole[:, i, j]) for j in range(output_dim_quadrupoles)] for i in range(output_dim_quadrupoles)]
r2_quadrupole = [[r2_score(actuals_quadrupole[:, i, j], preds_quadrupole[:, i, j]) for j in range(output_dim_quadrupoles)] for i in range(output_dim_quadrupoles)]

print(f'Dipole Model: MAE = {mae_dipole}, R^2 = {r2_dipole}')
print(f'Quadrupole Model: MAE = {mae_quadrupole}, R^2 = {r2_quadrupole}')