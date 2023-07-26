import torch
from torch_geometric.data import DataLoader
from egnn_pytorch import EGNN_Network
from QM93D_MM import QM93D
from sklearn.metrics import r2_score

# Function to calculate R2 score
def calculate_r2_score(y_true, y_pred):
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return r2_score(y_true, y_pred)

# Load data
dataset = QM93D(root='data')

# Combine pos and z to make a new feature vector
for data in dataset:
    data.x = torch.cat([data.pos, data.z.view(-1, 1)], dim=-1)

# Split data into train, validation and test sets
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define EGNN Network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = EGNN_Network(
    num_tokens = dataset[0].x.size(1),  # updated to match the dimension of the new feature vector
    dim = 32,
    depth = 3,
    num_nearest_neighbors = 8,
    coor_weights_clamp_value = 2.   # absolute clamped value for the coordinate weights, needed if you increase the num nearest neighbors
).to(device)

# Define loss function
criterion = torch.nn.MSELoss()

# Define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
for epoch in range(100):  # for simplicity, we just run 100 epochs
    for data in train_loader:
        data = data.to(device)
        mask = torch.ones_like(data.z).bool().to(device)
        optimizer.zero_grad()
        
        feats_out, coors_out = net(data.x, data.pos, mask=mask) 
        loss = criterion(feats_out.squeeze(), data.dipole)
        
        loss.backward()
        optimizer.step()

    # Validation loop
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            mask = torch.ones_like(data.z).bool().to(device)
            
            feats_out, coors_out = net(data.x, data.pos, mask=mask)
            val_loss = criterion(feats_out.squeeze(), data.dipole)
    
    print(f'Epoch: {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    # Save the model after each epoch
    torch.save(net.state_dict(), f'EGNN_model_epoch_{epoch+1}.pth')

# Test loop
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        mask = torch.ones_like(data.z).bool().to(device)
        
        feats_out, coors_out = net(data.x, data.pos, mask=mask)
        test_loss = criterion(feats_out.squeeze(), data.dipole)
        test_r2_score = calculate_r2_score(data.dipole, feats_out.squeeze())

print(f'Test Loss: {test_loss.item()}, Test R2 Score: {test_r2_score}')

