import numpy as np
import os
import os.path as osp

from spektral.data import Graph
from spektral.datasets.qm9 import QM9

class CustomQM9EdgeDataset(QM9):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dipoles_file = osp.join(self.path, 'dipoles.npy')
        self.quadrupoles_file = osp.join(self.path, 'quadrupoles.npy')

    def read(self):
        # Load the data using the base class's read method
        graphs = super().read()

        # Check if dipoles and quadrupoles have been saved to disk
        if osp.exists(self.dipoles_file) and osp.exists(self.quadrupoles_file):
            dipoles_all = np.load(self.dipoles_file)
            quadrupoles_all = np.load(self.quadrupoles_file)
        else:
            # Compute dipoles and quadrupoles and save them to disk
            dipoles_all = []
            quadrupoles_all = []
            for graph in graphs:
                dipoles, quadrupoles = self.compute_moments(graph)
                dipoles_all.append(dipoles)
                quadrupoles_all.append(quadrupoles)

            dipoles_all = np.array(dipoles_all)
            quadrupoles_all = np.array(quadrupoles_all)
            np.save(self.dipoles_file, dipoles_all)
            np.save(self.quadrupoles_file, quadrupoles_all)

        # Attach the dipoles and quadrupoles to the graph objects
        for i, graph in enumerate(graphs):
            graph.dipoles = dipoles_all[i]
            graph.quadrupoles = quadrupoles_all[i]

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
