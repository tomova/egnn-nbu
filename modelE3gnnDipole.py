import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.o3 import Irreps
from QM93D_MM import QM93D
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class E3nnModel(torch.nn.Module):
    def __init__(self):
        super(E3nnModel, self).__init__()

        self.fc_pos = FullyConnectedNet([3, 2])
        self.fc_z = FullyConnectedNet([1, 2])
        self.fc_out = FullyConnectedNet([2, 3])

    def forward(self, data):
        pos = F.relu(self.fc_pos(data.pos))
        z = F.relu(self.fc_z(data.z.unsqueeze(-1).float()))
        x = pos * z  # Element-wise multiplication
        x = self.fc_out(x)
        x = global_mean_pool(x, data.batch)  # Pooling operation on a per-graph basis
        return x
    
# Load data
dataset = QM93D(root='data')

for i in range(10):
    data = dataset[i]
    print(f'Data point {i}:', data)
    print('Dipole:', data.dipole, 'Shape:', data.dipole.shape)


# Split data into train, validation and test sets
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

# Define the E(3) equivariant GNN model
Rs_in = [(1, 0, 1)]
Rs_out = [(1, 0, 1)]
model = E3nnModel().to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.L1Loss()

# Define the data loaders for each set
train_loader = DataLoader(train_dataset, batch_size=564, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the evaluation function for both MAE and R2
def evaluate(model, data_loader):
    model.eval()
    total_abs_error = 0
    total_sq_error = 0
    total_variance = 0
    total_examples = 0
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            prediction = model(data)
            error = prediction - data.dipole  # Use data.dipole as target
            total_abs_error += torch.abs(error).sum().item()
            total_sq_error += torch.pow(error, 2).sum().item()
            total_variance += torch.var(data.dipole, unbiased=False).item() * (data.dipole.shape[0] - 1)  # Use data.dipole
            total_examples += data.dipole.shape[0]
    mae = total_abs_error / total_examples
    r2 = 1 - (total_sq_error / total_variance)
    return mae, r2

# Training loop
n_epochs = 1000
validation_interval = 20

for epoch in range(n_epochs):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        print("Output shape:", out.shape)
        print("Dipole shape:", batch.dipole.shape)
        loss = loss_func(out, batch.dipole)
        loss.backward()
        optimizer.step()

    if epoch % validation_interval == 0:
        train_mae, train_r2 = evaluate(model, train_loader)
        val_mae, val_r2 = evaluate(model, val_loader)
        print(f"Epoch: {epoch}, Train MAE: {train_mae}, Train R2: {train_r2}, Val MAE: {val_mae}, Val R2: {val_r2}")

# Evaluation
model.eval()
train_mae, train_r2 = evaluate(model, train_loader)
val_mae, val_r2 = evaluate(model, val_loader)
test_mae, test_r2 = evaluate(model, test_loader)
print(f"Train MAE: {train_mae}, Train R2: {train_r2}")
print(f"Validation MAE: {val_mae}, Validation R2: {val_r2}")
print(f"Test MAE: {test_mae}, Test R2: {test_r2}")
