
import torch
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
from e3nn import o3
from QM93D_MM import QM93D
import torch.nn.functional as F
from torch import nn
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.o3 import FullyConnectedTensorProduct, Linear
from torch_geometric.nn import MessagePassing
from torch_cluster import radius_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_node_attr, irreps_edge_attr, irreps_out, number_of_edge_features, radial_layers, radial_neurons, num_neighbors):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_out = o3.Irreps(irreps_out)
        self.num_neighbors = num_neighbors

        self.sc = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        self.lin1 = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)
        self.lin2 = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(number_of_edge_features, radial_neurons),
            torch.nn.SiLU(),
            torch.nn.Linear(radial_neurons, self.irreps_in.dim * self.irreps_edge_attr.dim * self.irreps_out.dim)
        )

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_features) -> torch.Tensor:
        x = node_input
        edge_features = self.fc(edge_features)
        edge_features = edge_features.view(-1, self.irreps_in.dim, self.irreps_edge_attr.dim, self.irreps_out.dim)

        s = self.sc(node_input, node_attr)
        x = self.lin1(node_input, node_attr)

        x = torch.einsum('nci,neio->nceo', x[edge_src], edge_features)
        x = scatter(x, edge_dst, dim=0, dim_size=node_input.shape[0])

        x = self.lin2(x.div(self.num_neighbors**0.5), node_attr)

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        m = self.sc.output_mask
        c_x = (1 - m) + c_x * m
        return c_s * s + c_x * x

class GatedConvModel(torch.nn.Module):
    def __init__(self, n_atom_basis=16, n_filters=32, n_gaussians=32, n_outputs=3, max_radius=1.0, h=0.2, cutoff=3.5):
        super(GatedConvModel, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters
        self.max_radius = max_radius
        self.cutoff = cutoff

        self.node_features = nn.Linear(3, n_atom_basis)

        self.filter_network = FullyConnectedNet(
            [n_atom_basis] + 3 * [n_filters] + [n_filters * n_gaussians],
        )

        irreps_gates = [(1, (0, 0))]  # gates are scalars
        irreps_out = n_atom_basis * [(1, (0, 0))]  # atom centered basis functions

        #self.gate = Gate("16x0o", [torch.tanh], "32x0o", [torch.sigmoid], "16x1e")
        self.gate = Gate("16x0e", [torch.tanh], "32x0e", [torch.sigmoid], "16x1e+16x1o")


        self.atom_wise = nn.Sequential(
            nn.BatchNorm1d(n_atom_basis),
            nn.Linear(n_atom_basis, n_outputs),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, data):
        x = self.node_features(data.pos)
        edge_index = radius_graph(data.pos, r=self.cutoff, batch=None, loop=True)

        # Ensure edge_attr is not None and has the correct size
        if data.edge_attr is None or data.edge_attr.shape[0] != edge_index.shape[1]:
            # Calculate pairwise distance for each edge
            pairwise_dist = torch.norm(data.pos[edge_index[0]] - data.pos[edge_index[1]], dim=-1, keepdim=True)
        else:
            # If edge_attr exists and has correct size, use it
            pairwise_dist = data.edge_attr

        # Edge embedding
        edge_emb = self.filter_network(pairwise_dist)

        x = self.gate(x, edge_index, edge_emb)
        out = self.atom_wise(x)
        return out


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
