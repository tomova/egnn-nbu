import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from egnn_pytorch import EGNN_Sparse_Network
from sklearn.metrics import r2_score
from QM93D_MM import QM93D


# Load data
dataset = QM93D(root='data')

# for data in dataset:
 #    data.dipole = data.dipole.view(1, 3)

# Split data into train, validation and test sets
split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
train_dataset = dataset[split_idx['train']]
val_dataset = dataset[split_idx['valid']]
test_dataset = dataset[split_idx['test']]


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
# model = EGNN_Sparse_Network(n_layers=4, feats_dim=9, pos_dim=3, edge_attr_dim=3, m_dim=16, fourier_features=0, soft_edge=0, 
#                             embedding_nums=[], embedding_dims=[], edge_embedding_nums=[], edge_embedding_dims=[], 
#                             update_coors=True, update_feats=True, norm_feats=True, norm_coors=False, 
 #                            norm_coors_scale_init=1e-2, dropout=0., coor_weights_clamp_value=None, 
 #                            aggr="add", global_linear_attn_every=0, global_linear_attn_heads=8, 
 #                            global_linear_attn_dim_head=64, num_global_tokens=4, recalc=0, output_dim=12).to(device)

# Initialize the model for predicting dipole moments
model_dipole = EGNN_Sparse_Network(n_layers=4, feats_dim=9, pos_dim=3, edge_attr_dim=3, m_dim=16, fourier_features=0, 
                                   soft_edge=0, embedding_nums=[], embedding_dims=[], edge_embedding_nums=[], 
                                   edge_embedding_dims=[], update_coors=True, update_feats=True, norm_feats=True, 
                                   norm_coors=False, norm_coors_scale_init=1e-2, dropout=0., coor_weights_clamp_value=None, 
                                   aggr="add", global_linear_attn_every=0, global_linear_attn_heads=8, 
                                   global_linear_attn_dim_head=64, num_global_tokens=4, recalc=0, output_dim=3).to(device)


# Initialize the optimizer and loss function
optimizer = torch.optim.Adam(model_dipole.parameters(), lr=0.001)
criterion = nn.MSELoss()  # MSE is appropriate for a regression problem

# Training loop for predicting dipole moments
for epoch in range(1000):
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        x = torch.cat([data.z, data.pos], dim=-1)
        out = model_dipole(x, data.edge_index, data.batch, data.edge_attr)
        #out = model_dipole(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = criterion(out, data.dipole)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch+1}, MSE Loss: {total_loss / len(train_loader)}")


    # Validation
    model_dipole.eval()
    with torch.no_grad():
        preds = []
        targets = []
        for data in valid_loader:
            data = data.to(device)
            x = torch.cat([data.z, data.pos], dim=-1)
            out = model_dipole(x, data.edge_index, data.batch, data.edge_attr)

            #out = model_dipole(data.x, data.edge_index, data.batch, data.edge_attr)
            preds.append(out.detach().cpu())
            targets.append(data.dipole.detach().cpu())

        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        # Calculate the R2 score
        r2 = r2_score(targets.numpy(), preds.numpy())
        print('Epoch: {:02d}, R2: {:.4f}'.format(epoch+1, r2))
    
    model_dipole.train()

# Save the model after training
torch.save(model_dipole.state_dict(), 'dipole_model.pth')

## Initialize the model
#model_dipole = EGNN_Sparse_Network(n_layers=4, feats_dim=9, pos_dim=3, edge_attr_dim=3, m_dim=16, fourier_features=0, 
#                                   soft_edge=0, embedding_nums=[], embedding_dims=[], edge_embedding_nums=[], 
#                                   edge_embedding_dims=[], update_coors=True, update_feats=True, norm_feats=True, 
#                                   norm_coors=False, norm_coors_scale_init=1e-2, dropout=0., coor_weights_clamp_value=None, 
#                                   aggr="add", global_linear_attn_every=0, global_linear_attn_heads=8, 
#                                   global_linear_attn_dim_head=64, num_global_tokens=4, recalc=0, output_dim=3).to(device)

# Load the weights
#model_dipole.load_state_dict(torch.load('dipole_model.pth'))

# model_dipole.eval()
# with torch.no_grad():
 #    for data in test_loader:
#         data = data.to(device)
#         predictions = model_dipole(data.x, data.edge_index, data.batch, data.edge_attr)
        # Do something with the predictions...

model_dipole.eval()
with torch.no_grad():
    preds = []
    targets = []
    total_test_loss = 0
    for data in test_loader:
        data = data.to(device)
        x = torch.cat([data.z, data.pos], dim=-1)
        out = model_dipole(x, data.edge_index, data.batch, data.edge_attr)
        loss = criterion(out, data.dipole)
        total_test_loss += loss.item()

        #out = model_dipole(data.x, data.edge_index, data.batch, data.edge_attr)
        preds.append(out.detach().cpu())
        targets.append(data.dipole.detach().cpu())

    print(f"Test MSE Loss: {total_test_loss / len(test_loader)}")

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)

    # Calculate the R2 score
    r2 = r2_score(targets.numpy(), preds.numpy())
    print('Test R2: {:.4f}'.format(r2))

