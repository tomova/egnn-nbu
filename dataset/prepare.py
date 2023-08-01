import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

# Function to calculate dipole moment
def calculate_dipole_moment(mol):
    conf = mol.GetConformer()
    dipole = [0.0, 0.0, 0.0]
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp('_GasteigerCharge'))
        position = conf.GetAtomPosition(atom.GetIdx())
        dipole[0] += charge * position.x
        dipole[1] += charge * position.y
        dipole[2] += charge * position.z
    return dipole

# Function to calculate quadrupole moment
def calculate_quadrupole_moment(mol):
    conf = mol.GetConformer()
    quadrupole = np.zeros((3, 3))
    for atom in mol.GetAtoms():
        charge = float(atom.GetProp('_GasteigerCharge'))
        position = conf.GetAtomPosition(atom.GetIdx())
        for i in range(3):
            for j in range(3):
                quadrupole[i, j] += charge * position[i] * position[j]
    return quadrupole

def mol_to_graph(mol):
    # Convert atoms to features (atomic numbers)
    atom_features = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    # Get adjacency matrix
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)

    # Get atom coordinates
    conformer = mol.GetConformer()
    atom_positions = [conformer.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()]
    atom_positions = [np.array([pos.x, pos.y, pos.z]) for pos in atom_positions]

    return np.array(atom_features), adjacency_matrix, atom_positions


# Load molecules from SDF file
sdf_supplier = Chem.SDMolSupplier('gdb9.sdf')

# List to store results
dataset = []

# Process each molecule
for mol in sdf_supplier:
    # Compute Gasteiger charges
    rdPartialCharges.ComputeGasteigerCharges(mol)

    # Convert molecule to graph representation
    atom_features, adjacency_matrix, atom_positions = mol_to_graph(mol)

    # Calculate and store dipole and quadrupole moments
    dipole = calculate_dipole_moment(mol)
    quadrupole = calculate_quadrupole_moment(mol)

    # Store graph representation and properties
    dataset.append((atom_features, adjacency_matrix, atom_positions, dipole, quadrupole))

# Save to disk
with open('dataset.pkl', 'wb') as file:
    pickle.dump(dataset, file)

# Load the data
# with open('qm9_data.pkl', 'rb') as f:
#     data = pickle.load(f)