import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import mean_squared_error
from QM93D_MM import QM93D

class QuadrupolePredictor(torch.nn.Module):
    def __init__(self):
        super(QuadrupolePredictor, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(4, 128)
        self.conv2 = GraphConv(128, 64)
        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 9)  # Predicting 9-dimensional quadrupole

    def forward(self, data):
        x = torch.cat([data.pos, data.z.view(-1, 1)], dim=-1)
        edge_attr = data.edge_attr
        edge_index = data.edge_index

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)

        return x

# Usage
model = QuadrupolePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load data
dataset = QM93D(root='data')

# Split data into train, validation, and test sets
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def train():
    model.train()
    for data in train_loader:
        y_true = data.quadrupole.view(-1, 9)  # Use quadrupole as the target
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, y_true)  # Mean Squared Error as the loss function
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        out = model(data)
        pred = out.detach()  # Detach the output from the computation graph
        y_true.append(data.quadrupole.view(-1, 9))
        y_pred.append(pred)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    mse = F.mse_loss(y_pred, y_true).item()
    return mse

# Train and validate the model
for epoch in range(1, 1001):  # Training for 1000 epochs
    train()
    train_mse = test(train_loader)
    val_mse = test(val_loader)
    print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}')

# After all epochs, test the model
test_mse = test(test_loader)
print(f'\nAfter {epoch} epochs, Test MSE: {test_mse:.4f}')

# Save the model
torch.save(model.state_dict(), 'quadrupole_predictor.pth')
