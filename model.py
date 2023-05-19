from QM93D_MM import QM93D

# Load the dataset
dataset = QM93D(root='data')

# Access the first data object in the dataset
data = dataset[0]

# Print the properties of the data object
print(data.pos)  # Positions of the atoms
print(data.z)  # Atomic numbers
print(data.y)  # Scalar properties
print(data.dipole)  # Dipole moment
print(data.quadrupole)  # Quadrupole moment