import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from sklearn.utils import shuffle

from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_undirected

# Approximate atomic weights
ATOMIC_WEIGHTS = {
    1: 1.008,  # Hydrogen
    6: 12.01,  # Carbon
    7: 14.007,  # Nitrogen
    8: 15.999,  # Oxygen
    9: 18.998,  # Fluorine
}

#mostly taken from https://github.com/divelab/DIG/blob/dig-stable/dig/threedgraph/dataset/PygQM93D.py 
class QM93D(InMemoryDataset):

    def __init__(self, root = 'dataset/', transform = None, pre_transform = None, pre_filter = None):

        self.url = 'https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz'
        self.folder = osp.join(root, 'qm9')

        super(QM93D, self).__init__(self.folder, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def calculate_center_of_mass(self, Rs, Zs):
        centers_of_mass = []
        for R, Z in zip(Rs, Zs):
            mass = np.array([ATOMIC_WEIGHTS[z] for z in Z])
            center_of_mass = np.sum(R * mass[:, None], axis=0) / np.sum(mass)
            centers_of_mass.append(center_of_mass)
        return np.array(centers_of_mass)

    def calculate_dipole_moments(self, Rs, Zs, centers_of_mass):
        dipole_moments = []
        for R, Z, center_of_mass in zip(Rs, Zs, centers_of_mass):
            R_centered = R - center_of_mass
            dipole_moment = np.sum(Z[:, None] * R_centered, axis=0)
            dipole_moments.append(dipole_moment)
        return np.array(dipole_moments)

    def calculate_quadrupole_moments(self, Rs, Zs, centers_of_mass):
        quadrupole_moments = []
        for R, Z, center_of_mass in zip(Rs, Zs, centers_of_mass):
            R_centered = R - center_of_mass
            quadrupole_moment = np.sum(Z[:, None, None] * np.einsum('ij,ik->ijk', R_centered, R_centered), axis=0)
            quadrupole_moments.append(quadrupole_moment)
        return np.array(quadrupole_moments)

    @property
    def raw_file_names(self):
        return 'qm9_eV.npz'

    @property
    def processed_file_names(self):
        return 'qm9_pyg.pt'

    def download(self):
        download_url(self.url, self.raw_dir)

    def add_edge_features(self, data):
        num_nodes = data.pos.size(0)
        device = data.pos.device
        edge_index = to_undirected(torch.combinations(torch.arange(num_nodes, device=device)))
        row, col = edge_index
        edge_attr = torch.norm(data.pos[row] - data.pos[col], p=2, dim=-1).view(-1, 1)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data

    def process(self):
        
        data = np.load(osp.join(self.raw_dir, self.raw_file_names))

        R = data['R']
        Z = data['Z']
        N= data['N']
        split = np.cumsum(N)
        R_qm9 = np.split(R, split)
        Z_qm9 = np.split(Z,split)
        target = {}
        for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']:
            target[name] = np.expand_dims(data[name],axis=-1)
        # y = np.expand_dims([data[name] for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']], axis=-1)
        
        centers_of_mass = self.calculate_center_of_mass(R_qm9, Z_qm9)
        dipoles = self.calculate_dipole_moments(R_qm9, Z_qm9, centers_of_mass)
        quadrupoles = self.calculate_quadrupole_moments(R_qm9, Z_qm9, centers_of_mass)

        data_list = []
        for i in tqdm(range(len(N))):
            R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
            z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
            y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']]
            dipole_i = torch.tensor(dipoles[i], dtype=torch.float32)
            quadrupole_i = torch.tensor(quadrupoles[i], dtype=torch.float32)
            data = Data(pos=R_i, z=z_i, y=y_i[0], mu=y_i[0], alpha=y_i[1], homo=y_i[2], lumo=y_i[3], gap=y_i[4], r2=y_i[5], zpve=y_i[6], U0=y_i[7], U=y_i[8], H=y_i[9], G=y_i[10], Cv=y_i[11], dipole=dipole_i, quadrupole=quadrupole_i)
            # Add edge features here:
            data = self.add_edge_features(data)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train':train_idx, 'valid':val_idx, 'test':test_idx}
        return split_dict

if __name__ == '__main__':
    dataset = QM93D()
    print(dataset)
    print(dataset.data.z.shape)
    print(dataset.data.pos.shape)
    target = 'mu'
    dataset.data.y = dataset.data[target]
    print(dataset.data.y.shape)
    print(dataset.data.y)
    print(dataset.data.mu)
    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    print(split_idx)
    print(dataset[split_idx['train']])
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))
    print(data)

# Load the dataset
# dataset = QM93D(root='data')

# Access the first data object in the dataset
# data = dataset[0]

# Print the properties of the data object
# print(data.pos)  # Positions of the atoms
# print(data.z)  # Atomic numbers
# print(data.y)  # Scalar properties
# print(data.dipole)  # Dipole moment
# print(data.quadrupole)  # Quadrupole moment