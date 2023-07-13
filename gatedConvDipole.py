import torch
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
from e3nn.nn import GatedConvParityNetwork
from QM93D_MM import QM93D
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GatedConvModel(torch.nn.Module):
    def __init__(self):
        super(GatedConvModel, self).__init__()

        self.node_features = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64)
        )

        self.edge_network = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64)
        )

        self.gnn = GatedConvParityNetwork(
            irreps_node_input="0e",
            irreps_node_output="1e",
            irreps_node_hidden="1e",
            irreps_edge_attr="1e",
            layers=4,
            max_radius=1.0,
            number_of_basis=3,
        )

        self.fc = torch.nn.Linear(64, 3) # to match the 3D dipole moment

    def forward(self, data):
        x = self.node_features(data.x.view(-1, 1))
        edge_attr = self.edge_network(data.edge_attr.view(-1, 1))
        out = self.gnn(x, data.edge_index, edge_attr)
        out = scatter_add(out, data.batch, dim=0)
        return self.fc(out)

# Load data
dataset = QM93D(root='data')

for data in dataset:
    data.dipole = data.dipole.view(1, 3)

# Split data into train, validation and test sets
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

# Define the GatedConv GNN model
model = GatedConvModel().to(device)


# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
loss_funcMSE = torch.nn.MSELoss()

# Define the data loaders for each set
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
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

# Initialize the patience counter
patience_counter = 0
patience_limit = 100

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

        # If the validation loss improved, save the model and reset the patience counter
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_modelE3D.pt")
            patience_counter = 0

        # If the validation loss did not improve, increment the patience counter
        else:
            patience_counter += 1

        # If the patience limit is reached, stop the training
        if patience_counter >= patience_limit:
            print("Early stopping due to no improvement in validation loss.")
            break

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
