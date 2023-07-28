import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.optim as optim
from torch_geometric.data import DataLoader
#from torch_geometric.datasets import QM9
from sklearn.metrics import r2_score
from QM93D_MM import QM93D
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def r2(loader, model):
    model.eval()
    preds = []
    targets = []
    for data in loader:
        data = data.to(device)
        preds.append(model(data))
        targets.append(data.dipole)
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    return r2_score(targets.cpu().numpy(), preds.cpu().detach().numpy())

class ModifiedEGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ModifiedEGNN, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # Compute messages
        x_j = self.lin(x_j)
        return x_j

    def update(self, aggr_out):
        # Update node embeddings
        return aggr_out

class ModifiedEGNN_Network(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super(ModifiedEGNN_Network, self).__init__()

        self.egnn = ModifiedEGNN(num_node_features, num_edge_features)
        self.fc1 = torch.nn.Linear(num_node_features, 128)
        self.fc2 = torch.nn.Linear(128, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if edge_index is None:
            edge_index = torch.tensor([[i, i] for i in range(x.shape[0])], dtype=torch.long).T

        # Pass node and edge features through the EGNN
        x = self.egnn(x, edge_index.to(x.device))

        # Pass through fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Sum up the node features to get a graph-level output
        x = torch.sum(x, dim=0)

        return x
    
def train(epoch, model, loader, optimizer):
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data.dipole)  # Use data.dipole instead of data.y
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)

def test(loader, model):
    model.eval()

    error = 0
    for data in loader:
        data = data.to(device)
        error += (model(data) - data.dipole).abs().sum().item()  # Use data.dipole instead of data.y
    return error / len(loader.dataset)

def main():
    # create a dataset
    dataset = QM93D(root='data')

    #for data in dataset:
    #    data.dipole = data.dipole.view(1, 3)

# Split data into train, validation and test sets
    split_idx = dataset.get_idx_split(len(dataset), train_size=110000, valid_size=10000, seed=42)
    train_dataset = dataset[split_idx['train']]
    val_dataset = dataset[split_idx['valid']]
    test_dataset = dataset[split_idx['test']]
    #dataset = QM9(root='/tmp/QM9')
    #dataset = dataset.shuffle()
    #train_dataset = dataset[:8000]
    #val_dataset = dataset[8000:9000]
    #test_dataset = dataset[9000:]
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize network with input and output sizes
    net = ModifiedEGNN_Network(3, 64).to(device)

    # Define optimizer and criterion
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    for epoch in range(100):
        loss = train(epoch, net, train_loader, optimizer)
        val_error = test(val_loader, net)
        val_r2 = r2(val_loader, net)
        print(f"Epoch: {epoch+1}, Training Loss: {loss}, Validation Error: {val_error}, Validation R^2: {val_r2}")

    # Test
    test_error = test(test_loader, net)
    test_r2 = r2(test_loader, net)
    print(f"Test Error: {test_error}, Test R^2: {test_r2}")


if __name__ == "__main__":
    main()