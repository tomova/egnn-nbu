from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import to_dense_adj
import pickle
import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from se3_transformer_pytorch import SE3Transformer
from torch.nn.functional import mse_loss
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, explained_variance_score, max_error


dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'dataset', 'datasetQM9.pkl')

with open(dataset_path, 'rb') as f:
    data = pickle.load(f)


max_num_nodes = 0
dataset = []
for atom_features, atom_positions, adjacency_matrix, bond_features, dipole, _ in data:
    edge_index = torch.tensor(np.stack(np.where(adjacency_matrix == 1)), dtype=torch.long)
    x = torch.tensor(atom_features, dtype=torch.float)
    num_nodes = x.shape[0]  # Number of nodes in the current graph
    max_num_nodes = max(max_num_nodes, num_nodes)  # Update if greater than previous max
    pos = torch.tensor(np.array(atom_positions), dtype=torch.float)
    y = torch.tensor(dipole, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index, pos=pos, y=y)
    dataset.append(graph_data)

print("Maximum number of atoms:", max_num_nodes)
# Define the split sizes
total_samples = len(dataset)
train_size = 110000
val_size = 10000
test_size = total_samples - train_size - val_size

# Split the data using train_test_split with specific random seed
train_data, temp_data = train_test_split(dataset, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)


# Print the size of each split
print("Train size:", len(train_data))
print("Validation size:", len(val_data))
print("Test size:", len(test_data))
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class GlobalSumPooling(nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


class DipolePredictorSE3(nn.Module):
    def __init__(self, output_dim=3, depth=1, hidden_dim=32):
        super(DipolePredictorSE3, self).__init__()
        # Define the parameters that suit your problem
        dim = 5
        input_degrees = 1

        self.se3_transformer = SE3Transformer(
            dim=dim,
            input_degrees=input_degrees,
            depth=depth,
            num_adj_degrees=2,
            attend_sparse_neighbors = True,
            num_neighbors = 0,
            num_degrees=depth + 1, # Assuming 3 is the desired number of layers
        )
        self.fc = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # A fully connected layer to produce the final output
        #self.fc = nn.Sequential(
         #   nn.Linear(5, hidden_dim // 2),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim // 2, output_dim)
      #  )

    def forward(self, feats, coors, adj_mat):
    #    print("Feats shape:", feats.shape) # Should be (batch_size, num_nodes, feature_dim)
     #   print("Coors shape:", coors.shape) # Check based on your requirements
      #  print("Adj_mat shape len:", len(adj_mat.shape)) # Check based on your requirements
        batch_size = feats.shape[0]
        nodes_size = feats.shape[1]
        mask  = torch.ones(batch_size, nodes_size).bool().to(device)
        x_transformed = self.se3_transformer(feats, coors, mask, adj_mat = adj_mat)
        # Create the mask from the adjacency matrix
 #       mask = (adj_mat.sum(dim=-1) != 0).float()
#        mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
   #     print("x_transformed 1 shape:", x_transformed.shape)
        # Now, x_transformed may depend on the specific API of SE3Transformer
#        x_transformed = self.se3_transformer(feats, coors, mask, adj_mat = adj_mat)
    #    print("x_transformed 2 shape:", x_transformed.shape)
        # Take the mean across the nodes, preserving batch and feature dims
        x_pooled = x_transformed.mean(dim=1)
     #   print("x_pooled shape:", x_pooled.shape)
        # Pass through the fully connected layers
        output = self.fc(x_pooled)

        return output


net = DipolePredictorSE3()
net.to(device)
for name, value in vars(net).items():
    print(f'{name}: {value}')


train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
#loss_function = nn.L1Loss()
loss_function_l1 = nn.L1Loss()
best_val_loss = float('inf')
for epoch in range(1000):
    net.train()
    print(f'Epoch {epoch}')
    total_loss_l1 = 0
    total_loss_mse = 0
    total_r2 = 0
    #total_loss = 0
    #total_r2 = 0
    for batch in train_loader:
        batch = batch.to(device)
        # Reshape the features and coordinates based on the batch vector
        feats, _ = to_dense_batch(batch.x, batch.batch) # Shape: (batch_size, num_nodes, num_features)
        coors, _ = to_dense_batch(batch.pos, batch.batch) # Shape: (batch_size, num_nodes, 3)
#        print(batch.x)
 #       print(batch.batch)
  #      print(batch)
        target = batch.y.view(-1, 3) # Shape: (batch_size, 3)

        batch_sizes = [torch.sum(batch.batch == i) for i in range(batch.batch.max() + 1)]
   #     print(batch_sizes)
        adj_mat = to_dense_adj(batch.edge_index, batch = batch.batch)
        feats_out = net(feats, coors, adj_mat=adj_mat)

        # Compute Loss
        #loss = loss_function(feats_out, target)
        # Compute Losses
        loss_l1 = loss_function_l1(feats_out, target)
        loss_mse = mse_loss(feats_out, target)
        # Backward propagation
        loss_l1.backward()
        # Backward propagation
        #loss.backward()
        # Update parameters
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()
        total_loss_l1 += loss_l1.item()
        total_loss_mse += loss_mse.item()
        r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
        total_r2 += r2
        #total_loss += loss.item()
        #r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
        #total_r2 += r2

    avg_loss_l1 = total_loss_l1 / len(train_loader)
    avg_r2 = total_r2 / len(train_loader)
    avg_loss_mse = total_loss_mse / len(train_loader)
    print(f'Epoch {epoch}, L1 Loss: {avg_loss_l1}, MSE Loss: {avg_loss_mse}, R2 Score: {avg_r2}')
    #avg_loss = total_loss / len(train_loader)
    #avg_r2 = total_r2 / len(train_loader)
    #print(f'Epoch {epoch}, Loss: {avg_loss}, R2 Score: {avg_r2}')

    # Validation
    net.eval()
    with torch.no_grad():
        val_loss_l1 = 0
        val_loss_mse = 0
        val_r2 = 0
        for batch in val_loader:
            batch = batch.to(device)
            feats, _ = to_dense_batch(batch.x, batch.batch)
            coors, _ = to_dense_batch(batch.pos, batch.batch)
            target = batch.y.view(-1, 3)
            adj_mat = to_dense_adj(batch.edge_index, batch=batch.batch)
            feats_out = net(feats, coors, adj_mat=adj_mat)

            # Compute Loss
            loss_l1 = loss_function_l1(feats_out, target)
            loss_mse = mse_loss(feats_out, target)
            val_loss_l1 += loss_l1.item()
            val_loss_mse += loss_mse.item()
            # Compute R2 score
            r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
            val_r2 += r2
        avg_val_loss_l1 = val_loss_l1 / len(val_loader)
        avg_val_loss_mse = val_loss_mse / len(val_loader)
        avg_val_r2 = val_r2 / len(val_loader)
        print(f'Validation L1 Loss: {avg_val_loss_l1}, Validation MSE Loss: {avg_val_loss_mse}, Validation R2 Score: {avg_val_r2}')



        # Save the model if it has the best validation loss so far
        if avg_val_loss_l1 < best_val_loss:
            best_val_loss = avg_val_loss_l1
            torch.save(net.state_dict(), 'best_model_se3_dipole_mae.pth')


# Testing
net.load_state_dict(torch.load('best_model_se3_dipole_mae.pth'))
net.eval()
with torch.no_grad():
    test_loss_l1 = 0
    test_loss_mse = 0
    test_r2 = 0
    test_rmse = 0
    test_mape = 0
    test_evs = 0
    test_me = 0
    for batch in test_loader:
        batch = batch.to(device)
        feats, _ = to_dense_batch(batch.x, batch.batch)
        coors, _ = to_dense_batch(batch.pos, batch.batch)
        target = batch.y.view(-1, 3)
        adj_mat = to_dense_adj(batch.edge_index, batch=batch.batch)
        feats_out = net(feats, coors, adj_mat=adj_mat)

        ## Compute Loss
        loss_l1 = loss_function_l1(feats_out, target)
        test_loss_l1 += loss_l1.item()
        loss_mse = mse_loss(feats_out, target)
        test_loss_mse += loss_mse.item()

        # Compute R2 score
        r2 = r2_score(target.cpu().numpy(), feats_out.detach().cpu().numpy())
        test_r2 += r2

        # Calculate additional metrics
        true_values = target.cpu().numpy()
        pred_values = feats_out.detach().cpu().numpy()
        test_rmse += np.sqrt(mean_squared_error(true_values, pred_values))
        test_mape += mean_absolute_percentage_error(true_values, pred_values)
        test_evs += explained_variance_score(true_values, pred_values)
        test_me += max_error(true_values, pred_values)

    avg_test_loss_l1 = test_loss_l1 / len(test_loader)
    avg_test_r2 = test_r2 / len(test_loader)
    avg_test_loss_mse = test_loss_mse / len(val_loader)
    avg_test_rmse = test_rmse / len(test_loader)
    avg_test_mape = test_mape / len(test_loader)
    avg_test_evs = test_evs / len(test_loader)
    avg_test_me = test_me / len(test_loader)
    print(f'Test Loss L1: {avg_test_loss_l1}, MSE Loss: {avg_test_loss_mse}, Test R2 Score: {avg_test_r2}')
    print(f'Test RMSE: {avg_test_rmse}, MAPE: {avg_test_mape}, Explained Variance Score: {avg_test_evs}, Max Error: {avg_test_me}')

# You can load the saved model using `net.load_state_dict(torch.load('best_model.pth'))`
# Then, you can pass your input features to the model using `net(feats, coors, adj_mat=adj_mat)`
# to get the predictions for the dipole moment.