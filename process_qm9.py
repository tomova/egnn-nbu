import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def calculate_quadrupole_moments(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) != -1:  # Check if the conformer generation is successful
            AllChem.ComputeGasteigerCharges(mol)
            coords = mol.GetConformer().GetPositions()
            charges = [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms()]
            
            # Calculate quadrupole moment tensor
            q_tensor = np.zeros((3, 3))
            for i in range(mol.GetNumAtoms()):
                r = coords[i]
                q = charges[i]
                for m in range(3):
                    for n in range(3):
                        if m == n:
                            q_tensor[m, n] += q * (3 * r[m] * r[n] - np.linalg.norm(r)**2)
                        else:
                            q_tensor[m, n] += q * (3 * r[m] * r[n])
            
            return q_tensor
    return None

def main():
    # Load the qm9.csv dataset
    df = pd.read_csv("data/qm9.csv")

    # Ensure SMILES validity
    df = df[df["smiles"].apply(is_valid_smiles)]

    # Calculate quadrupole moments and add as a new column
    df["quadrupole_moments"] = df["smiles"].apply(calculate_quadrupole_moments)

    # Save the updated dataframe to qm9_updated.csv
    df.to_csv("data/qm9_updated.csv", index=False)

if __name__ == "__main__":
    main()