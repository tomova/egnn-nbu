import torch
from torch_geometric.data import DataLoader
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct
from QM93D_MM import QM93D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class E3nnModel(torch.nn.Module):
    def __init__(self):
        super(E3nnModel, self).__init__()

        irreps_in = [(1, (2, 1))]
        irreps_in2 = [(1, (0, 1))]
        irreps_out = [(1, (1, 1))]
        self.tp = FullyConnectedTensorProduct(irreps_in, irreps_in2, irreps_out, shared_weights=False)

        irreps_hidden = [2, 2]
        self.fc = FullyConnectedNet([3] + irreps_hidden + [3]) 

    def forward(self, data):
        pos = data.pos.unsqueeze(0)  # Add batch dimension
        z = data.z.float().unsqueeze(-1).unsqueeze(0)  # Add batch dimension
        x = self.tp(pos, z)
        x = x.squeeze(0)  # Remove batch dimension
        x = self.fc(x)
        return x.sum(dim=-1)
    
# Load data
dataset = QM93D(root='data')

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
            error = prediction - data.y
            total_abs_error += torch.abs(error).sum().item()
            total_sq_error += torch.pow(error, 2).sum().item()
            total_variance += torch.var(data.y, unbiased=False).item() * (data.y.shape[0] - 1)  # Multiply by (n - 1) because torch.var divides by n
            total_examples += data.y.shape[0]
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
        loss = loss_func(out, batch.y)
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
