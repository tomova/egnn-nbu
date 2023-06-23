import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct
from e3nn.o3 import Irreps
from e3nn.nn import Gate
from QM93D_MM import QM93D
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class E3nnModel(torch.nn.Module):
    def __init__(self):
        super(E3nnModel, self).__init__()
        self.irreps_in = Irreps("1x0e + 1x1o")
        self.irreps_hidden = Irreps("2x0e + 2x1o + 1x2e")
        self.irreps_out = Irreps("1x0e")

        self.tp = FullyConnectedTensorProduct(self.irreps_in, self.irreps_in, self.irreps_hidden)
        self.fc = torch.nn.Linear(self.irreps_hidden.dim, self.irreps_out.dim)
        self.gate = Gate(self.irreps_hidden, self.irreps_hidden, self.irreps_out)


    def forward(self, data):
        x = torch.cat([data.z.unsqueeze(-1).float(), data.pos], dim=-1)
        x = self.tp(x, x)
        x = F.relu(x)
        x = self.fc(x)
        x = self.gate(x)
        x = global_mean_pool(x, data.batch)  # Pooling operation on a per-graph basis
        return x

# Continue from your existing code...
# Load data
dataset = QM93D(root='data')

for data in dataset:
    data.dipole = data.dipole.view(1, 3)

# Split data into train, validation and test sets
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

# Define the E(3) equivariant GNN model
Rs_in = [(1, 0, 1)]
Rs_out = [(1, 0, 1)]
# Define the E(3) equivariant GNN model
model = E3nnModel().to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
loss_funcMSE = torch.nn.MSELoss()
loss_funcL1 = torch.nn.L1Loss()

# Define the data loaders for each set
train_loader = DataLoader(train_dataset, batch_size=564, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the evaluation function for both MAE, MSE and R2
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
            target = data.dipole.view(-1, 3)  # Reshape target
            error = prediction - target  # Use reshaped data.dipole as target
            total_abs_error += torch.abs(error).sum().item()
            total_sq_error += torch.pow(error, 2).sum().item()
            total_variance += torch.var(target, unbiased=False).item() * (target.shape[0] - 1)  # Use reshaped data.dipole
            total_examples += target.shape[0]
    mae = total_abs_error / total_examples
    mse = total_sq_error / total_examples
    r2 = 1 - (total_sq_error / total_variance)
    return mae, mse, r2



# Training loop
n_epochs = 1000
validation_interval = 20
best_val_loss = float('inf')

for epoch in range(n_epochs):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        target = batch.dipole.view(-1, 3)
        loss = loss_funcMSE(out, target)
        loss.backward()
        optimizer.step()

    if epoch % validation_interval == 0:
        val_mae, val_mse, val_r2 = evaluate(model, val_loader)
        print(f"Epoch: {epoch}, Val MAE: {val_mae}, Val MSE: {val_mse}, Val R2: {val_r2}, Best Val Loss: {best_val_loss}")
        val_loss = val_mse # Use MSE for validation loss
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_modelE3D.pt")

# Load the best model
model.load_state_dict(torch.load("best_modelE3D.pt"))
#-------------------
    

# Evaluation
model.eval()
train_mae, train_mse, train_r2 = evaluate(model, train_loader)
val_mae, val_mse, val_r2 = evaluate(model, val_loader)
test_mae, test_mse, test_r2 = evaluate(model, test_loader)
print(f"Train MAE: {train_mae}, Train MSE: {train_mse}, Train R2: {train_r2}")
print(f"Validation MAE: {val_mae}, Validation MSE: {val_mse}, Validation R2: {val_r2}")
print(f"Test MAE: {test_mae}, Test MSE: {test_mse}, Test R2: {test_r2}")
