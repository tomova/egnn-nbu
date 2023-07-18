import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.metrics import r2_score
from se3_transformer_pytorch import SE3Transformer
from QM93D_MM import QM93D

class DipolePredictorSE3(torch.nn.Module):
    def __init__(self):
        super(DipolePredictorSE3, self).__init__()
        torch.manual_seed(12345)

        self.se3_transformer = SE3Transformer(
            dim=4,
            heads=8,
            depth=2,
            num_degrees=4,
            reduce_dim=True
        )
        
        self.fc = torch.nn.Linear(4, 3)

    def forward(self, data):
        x = self.se3_transformer(data.x, data.edge_index)
        x = x.mean(dim=0)  # reduce the output to a single vector
        x = self.fc(x)
        return x


# Usage
model = DipolePredictorSE3()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load data
dataset = QM93D(root='data')

# Same splitting as before
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

# Adjust data to SE3 Transformer
for data in train_dataset + val_dataset + test_dataset:
    data.x = torch.cat([data.pos, data.z.view(-1, 1)], dim=-1)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Same training function
def train():
    model.train()
    for data in train_loader:
        y_true = data.dipole.view(-1, 3)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, y_true)  
        loss.backward()
        optimizer.step()

# Same test function
def test(loader):
    model.eval()
    y_true = []
    y_pred = []
    for data in loader:
        out = model(data)
        pred = out.detach()  
        y_true.append(data.dipole.view(-1, 3))
        y_pred.append(pred)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    mse = F.mse_loss(y_pred, y_true).item()
    r2 = r2_score(y_true.numpy(), y_pred.numpy())
    return mse, r2

# Training and validating the model
for epoch in range(1, 1001):  
    train()
    train_mse, train_r2 = test(train_loader)
    val_mse, val_r2 = test(val_loader)
    print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Train R2: {train_r2:.4f}, Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}')

# After all epochs, test the model
test_mse, test_r2 = test(test_loader)
print(f'\nAfter {epoch} epochs, Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}')

# Save the model
torch.save(model.state_dict(), 'dipole_predictor_se3.pth')

# Load the model for future use
# loaded_model = DipolePredictorSE3()
# loaded_model.load_state_dict(torch.load('dipole_predictor_se3.pth'))
