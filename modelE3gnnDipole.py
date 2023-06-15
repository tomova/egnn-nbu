import torch
from torch_geometric.data import DataLoader
from torch.nn import MSELoss
from e3nn.networks import GatedConvParityNetwork
from QM93D_MM import QM93D
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
dataset = QM93D(root='data')
dataset = dataset.to(device)

# Split data into train, validation and test sets
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

# Define the E(3) equivariant GNN model
Rs_in = [(1, 0, 1)]
Rs_out = [(1, 0, 1)]
model = GatedConvParityNetwork(Rs_in, Rs_out, mul=6, lmax=2, layers=3)
model = model.to(device)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = MSELoss()

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
