import numpy as np
import os

from spektral.data import Graph
from spektral.datasets.qm9 import QM9

class CustomQM9EdgeDataset(QM9):
    def read(self):
        graphs = super().read()

        # Check if dipole and quadrupole moments are already saved
        dipole_file = os.path.join(self.path, "dipoles.npy")
        quadrupole_file = os.path.join(self.path, "quadrupoles.npy")
        
        if os.path.exists(dipole_file) and os.path.exists(quadrupole_file):
            dipoles = np.load(dipole_file)
            quadrupoles = np.load(quadrupole_file)
        else:
            # Compute dipole and quadrupole moments
            dipoles, quadrupoles = [], []
            for graph in graphs:
                dipole, quadrupole = self.compute_moments(graph)
                dipoles.append(dipole)
                quadrupoles.append(quadrupole)
            
            # Save to disk
            np.save(dipole_file, dipoles)
            np.save(quadrupole_file, quadrupoles)

        return graphs

    def compute_moments(self, graph):
        # Assuming atomic_num is at index 0 and coordinates at indices 5:8 and charge at index 8
        charges = graph.x[:, 8]
        coords = graph.x[:, 5:8]

        # Dipole moment calculation
        dipole = np.sum(charges[:, None] * coords, axis=0)

        # Quadrupole moment calculation
        quadrupole = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                quadrupole[i, j] = np.sum(charges * (3 * coords[:, i] * coords[:, j] - np.linalg.norm(coords, axis=1)**2 * (i == j)))

        return dipole, quadrupole
