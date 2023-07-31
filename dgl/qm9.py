from dgl.data import QM9EdgeDataset
import numpy as np
import torch
import dgl
from dgl.data import QM9EdgeDataset
from torch.utils.data import DataLoader
from egnn_pytorch import EGNN_Network
from torch import nn, optim
from sklearn.model_selection import train_test_split
from egnn_pytorch import EGNN
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class CustomQM9EdgeDataset(QM9EdgeDataset):
    def __init__(self, *args, **kwargs):
        super(CustomQM9EdgeDataset, self).__init__(*args, **kwargs)
        self.process_dipole_quadrupole_moments()

    def process_dipole_quadrupole_moments(self):
        for idx in range(len(self)):
            g, label = self[idx]

            # Retrieve coordinates and charges here, for example
            coordinates = g.ndata['pos']
            charges = g.ndata['charge']

            # Call calculation functions with those arguments
            dipole = self.calculate_dipole(coordinates, charges)
            quadrupole = self.calculate_quadrupole(coordinates, charges)

            # Add them to the node data of the graph
            g.ndata["dipole"] = dipole
            g.ndata["quadrupole"] = quadrupole

    # Additional functions to calculate dipole and quadrupole moments
    @staticmethod
    def calculate_dipole(coordinates, charges):
        return np.sum(coordinates * charges[:, np.newaxis], axis=0)

    @staticmethod
    def calculate_quadrupole(coordinates, charges):
        return np.sum([charge * np.outer(coord, coord) for charge, coord in zip(charges, coordinates)], axis=0)

# Load dataset
dataset = CustomQM9EdgeDataset()
print(dataset)
print(dataset[0])

# Split into training, validation, and test sets
train_idx, test_val_idx = train_test_split(np.arange(len(dataset)), test_size=20000, random_state=42)
test_idx, val_idx = train_test_split(test_val_idx, test_size=10000, random_state=42)

train_set = [dataset[i] for i in train_idx]
test_set = [dataset[i] for i in test_idx]
val_set = [dataset[i] for i in val_idx]

# Create data loaders
train_loader = DataLoader(train_set, batch_size=32, collate_fn=dgl.batch)
test_loader = DataLoader(test_set, batch_size=32, collate_fn=dgl.batch)
val_loader = DataLoader(val_set, batch_size=32, collate_fn=dgl.batch)

# Define the model
class DipoleEGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.egnn = EGNN_Network(
            # Add the necessary configurations
        )
        self.fc = nn.Linear(in_features=64, out_features=3) # Assuming 64 features from EGNN

    def forward(self, g):
        x = g.ndata['pos'] # Using position as an input feature
        x = self.egnn(x, g.edges())
        x = self.fc(x)
        return x
    
class CustomEGNNNetwork(EGNN_Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lin1 = torch.nn.Linear(kwargs['num_tokens'], kwargs['dim'])
    
    def forward(self, feats, coors, mask = None):
        feats = self.lin1(feats)

        # replace original forward method
        b, n, _ = coors.shape

        if mask is not None:
            mask = mask.bool()

        # initial features and coordinates
        x = feats

        # layer norm on input
        x = self.norm_in(x)

        for step in range(self.depth):
            # get neighbors and extend input
            rel_coors = get_relative_positions(coors = coors, num_nearest = self.num_nearest_neighbors)
            edge_index, _ = radius_graph(coors, r = self.r, batch = None, max_num_neighbors = self.num_nearest_neighbors)

            x, coors = self.layers[step](
                x,
                coors,
                edge_index,
                rel_coors,
                mask = mask
            )

        # final mlp head
        return self.mlp_head(x), coors



num_tokens = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomEGNNNetwork(
    num_tokens = num_tokens,  # updated to match the dimension of the new feature vector
    dim = 32,
    depth = 3 # absolute clamped value for the coordinate weights, needed if you increase the num nearest neighbors
).to(device)

    
#model = DipoleEGNN().to(device)


# Loss and optimizer
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100): # Number of epochs
    model.train()
    total_loss = 0
    total_r2 = 0
    num_batches = 0
    for batched_graph in train_loader:
        dipole = batched_graph.ndata['dipole']
        prediction = model(batched_graph)
        loss = loss_func(prediction, dipole)
        total_loss += loss.item()
        total_r2 += r2_score(dipole.detach().numpy(), prediction.detach().numpy())
        num_batches += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: MSE: {total_loss/num_batches}, R²: {total_r2/num_batches}")

# Validation code can be similar to the training loop

# Testing
model.eval()
total_loss = 0
total_r2 = 0
num_batches = 0
for batched_graph in test_loader:
    with torch.no_grad():
        dipole = batched_graph.ndata['dipole']
        prediction = model(batched_graph)
        loss = loss_func(prediction, dipole)
        total_loss += loss.item()
        total_r2 += r2_score(dipole.detach().numpy(), prediction.detach().numpy())
        num_batches += 1

print(f"Test Results - MSE: {total_loss/num_batches}, R²: {total_r2/num_batches}")

# Save the model
torch.save(model.state_dict(), "dipole_model.pth")

# To use the saved model later, load it using the following code:
# model = DipoleEGNN()
# model.load_state_dict(torch.load("dipole_model.pth"))
# model.eval()