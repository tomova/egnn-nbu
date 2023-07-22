import torch
import numpy as np
from torch.nn import Linear
from egnn_pytorch import EGNN_Sparse_Network
from torch.optim import Adam
from sklearn.metrics import r2_score
from torch.nn import MSELoss
from QM93D_MM import QM93D
from torch_geometric.loader import DataLoader
import torch.nn.functional as F 


class GNNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers):
        super(GNNModel, self).__init__()
        self.egnn = EGNN_Sparse_Network(
            feats_dim=num_features, 
            pos_dim=3, 
            edge_attr_dim=1, 
            m_dim=hidden_channels, 
            fourier_features=0,
            embedding_dim=hidden_channels,
            num_layers=num_layers,
        )
        self.lin1 = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, pos):
        x, pos = self.egnn(x, pos, edge_index, edge_attr)
        out = self.lin1(x)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load data
dataset = QM93D(root='data')
# Split data into train, validation and test sets
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
N = len(np.unique([atom_type for data in dataset for atom_type in data.z]))  # Total unique atom types

num_features = N + 3  # N one-hot features for atomic types and 3 for positions

model_dipole = GNNModel(num_features=num_features, hidden_channels=64, num_classes=3, num_layers=3)
model_dipole = model_dipole.to(device)

model_quadrupole = GNNModel(num_features=num_features, hidden_channels=64, num_classes=9, num_layers=3)
model_quadrupole = model_quadrupole.to(device)


optimizer_dipole = Adam(model_dipole.parameters(), lr=0.001)
optimizer_quadrupole = Adam(model_quadrupole.parameters(), lr=0.001)

loss_func = MSELoss()

def train(model, optimizer, target):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        data.z = F.one_hot(data.z, num_classes=N).float().to(device)  # One-hot encoding
        node_features = torch.cat([data.z, data.pos], dim=-1)  # Concatenate atomic numbers and positions
        optimizer.zero_grad()
        out = model(node_features, data.edge_index, data.edge_attr)
        loss = loss_func(out, getattr(data, target))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(loader, model, target):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.z = F.one_hot(data.z, num_classes=N).float().to(device)  # One-hot encoding
            node_features = torch.cat([data.z, data.pos], dim=-1)  # Concatenate atomic numbers and positions
            out = model(node_features, data.edge_index, data.edge_attr)
            loss = loss_func(out, getattr(data, target))
            total_loss += loss.item()
    return total_loss / len(loader)

def test(loader, model, target):
    model.eval()
    true_values = []
    predictions = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.z = F.one_hot(data.z, num_classes=N).float().to(device)  # One-hot encoding
            node_features = torch.cat([data.z, data.pos], dim=-1)  # Concatenate atomic numbers and positions
            out = model(node_features, data.edge_index, data.edge_attr)
            true_values.append(getattr(data, target).cpu())
            predictions.append(out.cpu())
    true_values = torch.cat(true_values)
    predictions = torch.cat(predictions)
    mse = loss_func(predictions, true_values).item()
    r2 = r2_score(true_values.numpy(), predictions.numpy())
    return mse, r2


# Training loop
for epoch in range(1, 101):
    loss_dipole = train(model_dipole, optimizer_dipole, 'dipole')
    val_loss_dipole = validate(val_loader, model_dipole, 'dipole')

    loss_quadrupole = train(model_quadrupole, optimizer_quadrupole, 'quadrupole')
    val_loss_quadrupole = validate(val_loader, model_quadrupole, 'quadrupole')

    print(f"Epoch: {epoch}, Dipole Loss: {loss_dipole:.4f}, Quadrupole Loss: {loss_quadrupole:.4f}")
    print(f"Dipole Val Loss: {val_loss_dipole:.4f}, Quadrupole Val Loss: {val_loss_quadrupole:.4f}")

# Testing
test_mse_dipole, test_r2_dipole = test(test_loader, model_dipole, 'dipole')
test_mse_quadrupole, test_r2_quadrupole = test(test_loader, model_quadrupole, 'quadrupole')

print(f"Dipole Test MSE: {test_mse_dipole:.4f}, Dipole Test R2: {test_r2_dipole:.4f}")
print(f"Quadrupole Test MSE: {test_mse_quadrupole:.4f}, Quadrupole Test R2: {test_r2_quadrupole:.4f}")

torch.save(model_dipole.state_dict(), 'model_dipole_egnn_pytorch')
torch.save(model_quadrupole.state_dict(), 'model_quadrupole_egnn_pytorch')

# For future use
# model_dipole = GNNModel(num_features=num_features, hidden_channels=64, num_classes=3, num_layers=3)
# model_dipole.load_state_dict(torch.load('model_dipole.pth'))
# model_dipole = model_dipole.to(device)
# model_dipole.eval()

# model_quadrupole = GNNModel(num_features=num_features, hidden_channels=64, num_classes=9, num_layers=3)
# model_quadrupole.load_state_dict(torch.load('model_quadrupole.pth'))
# model_quadrupole = model_quadrupole.to(device)
# model_quadrupole.eval()
