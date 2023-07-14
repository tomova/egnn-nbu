import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import r2_score

class DipolePredictor(torch.nn.Module):
    def __init__(self):
        super(DipolePredictor, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(4, 128)
        self.conv2 = GraphConv(128, 64)
        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 3)  # Predicting 3-dimensional dipole

    def forward(self, data):
        x = torch.cat([data.pos, data.z.view(-1, 1)], dim=-1)  # Concatenate positional and atomic number information
        edge_attr = data.edge_attr  # Bond lengths
        edge_index = data.edge_index

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)  # Global pooling
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x.view(-1, 3)  # Reshaping the output to match the dipole shape (n, 3)

# Usage
model = DipolePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    for data in train_loader:
        # Use dipole as target
        y_true = data.dipole
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, y_true)  # Mean Squared Error as loss function
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        out = model(data)
        pred = out.detach()  # Detaching the output from computation graph
        y_true.append(data.dipole)
        y_pred.append(pred)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    mse = F.mse_loss(y_pred, y_true).item()
    r2 = r2_score(y_true.numpy(), y_pred.numpy())
    return mse, r2

# Train and validate the model
for epoch in range(1, 1001):  # Training for 1000 epochs
    train()
    train_mse, train_r2 = test(train_loader)
    val_mse, val_r2 = test(val_loader)
    print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Train R2: {train_r2:.4f}, Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}')

# After all epochs, test the model
test_mse, test_r2 = test(test_loader)
print(f'\nAfter {epoch} epochs, Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}')

# Save the model
torch.save(model.state_dict(), 'dipole_predictor.pth')

# Load the model for future use
# loaded_model = DipolePredictor()
# loaded_model.load_state_dict(torch.load('dipole_predictor.pth'))

