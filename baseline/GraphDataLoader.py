import pandas as pd
from rdkit import Chem
import networkx as nx
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

class GraphDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.features_train = None
        self.features_test = None
        self.targets_train = None
        self.targets_test = None

    def load_dataset(self):
        self.df = pd.read_csv(self.dataset_path)

    def mol_to_pyg(self, mol):
        G = nx.Graph()

        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append([
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic())
            ])

        bond_indices = []
        for bond in mol.GetBonds():
            bond_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bond_indices.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])  # add reverse direction

        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(bond_indices, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)

        return data


    def convert_smiles_to_graphs(self):
        self.df["Graph"] = self.df["smiles"].apply(lambda x: self.mol_to_pyg(Chem.MolFromSmiles(x)))

    def add_charges_as_node_attributes(self):
        for i, row in self.df.iterrows():
            for j, node in enumerate(row["Graph"].nodes(data=True)):
                node[1]["charge"] = row["charges"][j]

    def prepare_data(self):
        self.convert_smiles_to_graphs()
        self.add_charges_as_node_attributes()

        self.features_train, self.features_test, self.targets_train, self.targets_test = train_test_split(
            self.df["Graph"].tolist(),
            self.df[["dipole", "quadrupole_moments"]].values,
            test_size=0.2,
            random_state=42
        )


