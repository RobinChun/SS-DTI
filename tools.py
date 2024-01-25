import random
import torch
import numpy as np
import re
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from torch.utils.data import Dataset


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "T": 20, "W": 3, "V": 13, "Y": 19, "X": 21, "U": 22}

CHARPROTLEN = 22


def get_kfold_data(i, datasets, k=5):
    fold_size = len(datasets) // k
    datasets = list(datasets)

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        valid_set = datasets[val_start:val_end]
        train_set = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        valid_set = datasets[val_start:val_end]
        train_set = datasets[val_end:]
    else:
        valid_set = datasets[val_start:]
        train_set = datasets[0:val_start]

    train_set = np.array(train_set)
    valid_set = np.array(valid_set)

    return train_set, valid_set


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def contact_padding(adjacency, dim):
    # Assuming node_features is your existing node features tensor with shape (num_nodes, feature_dim)
    adjacency = torch.tensor(adjacency, dtype=torch.float)
    num_nodes = adjacency.shape[0]
    if num_nodes < dim:
        padding_size = dim - num_nodes
        # Create padding matrix
        padding_matrix_1 = torch.zeros(num_nodes, padding_size)
        padding_matrix_2 = torch.zeros(padding_size, dim)

        # Pad adjacency matrix
        padded_adjacency = torch.cat([adjacency, padding_matrix_1], dim=1)
        padded_adjacency = torch.cat([padded_adjacency, padding_matrix_2], dim=0)
    else:
        padded_adjacency = adjacency[:dim, :dim]
    return padded_adjacency


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [
                  atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])  # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + np.eye(adjacency.shape[0])


def weighted_adjacency_matrix(mol):
    Chem.Kekulize(mol)
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    # print(adj_matrix)
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            bond_value = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_value = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_value = 3

        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx = bond.GetEndAtom().GetIdx()

        # Update the adjacency matrix with the bond value
        adj_matrix[begin_atom_idx][end_atom_idx] = bond_value
        adj_matrix[end_atom_idx][begin_atom_idx] = bond_value
    return np.array(adj_matrix) + np.eye(adj_matrix.shape[0])


def mol_features(mol):
    atom_feat = np.zeros((mol.GetNumAtoms(), 34))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix


def mol_features_weighted(mol):
    atom_feat = np.zeros((mol.GetNumAtoms(), 34))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = weighted_adjacency_matrix(mol)
    return atom_feat, adj_matrix


def graph_padding(node_features, adjacency, dim):
    # Assuming node_features is your existing node features tensor with shape (num_nodes, feature_dim)
    node_features = torch.tensor(node_features, dtype=torch.float)
    adjacency = torch.tensor(adjacency, dtype=torch.float)
    feature_dim = node_features.shape[1]
    num_nodes = node_features.shape[0]
    if num_nodes < dim:
        padding_size = dim - num_nodes

        # Create padding vectors
        padding_vectors = torch.zeros(padding_size, feature_dim)

        # Pad node_features tensor
        padded_node_features = torch.cat([node_features, padding_vectors], dim=0)

        # Create padding matrix
        padding_matrix_1 = torch.zeros(num_nodes, padding_size)
        padding_matrix_2 = torch.zeros(padding_size, dim)

        # Pad adjacency matrix
        padded_adjacency = torch.cat([adjacency, padding_matrix_1], dim=1)
        padded_adjacency = torch.cat([padded_adjacency, padding_matrix_2], dim=0)
    else:
        padded_node_features = node_features[:dim, :]
        padded_adjacency = adjacency[:dim, :dim]
    return padded_node_features, padded_adjacency


def contact_padding(adjacency, dim):
    # Assuming node_features is your existing node features tensor with shape (num_nodes, feature_dim)
    adjacency = torch.tensor(adjacency, dtype=torch.float)
    num_nodes = adjacency.shape[0]
    if num_nodes < dim:
        padding_size = dim - num_nodes
        # Create padding matrix
        padding_matrix_1 = torch.zeros(num_nodes, padding_size)
        padding_matrix_2 = torch.zeros(padding_size, dim)

        # Pad adjacency matrix
        padded_adjacency = torch.cat([adjacency, padding_matrix_1], dim=1)
        padded_adjacency = torch.cat([padded_adjacency, padding_matrix_2], dim=0)
    else:
        padded_adjacency = adjacency[:dim, :dim]
    return padded_adjacency
