
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from serenityff.charge.tree.dash_tree import DASHTree
from rdkit import Chem
import torch
# from custom_featurization_stuff import get_graph_from_mol
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

allowable_set= ["C","N","O","F","P","S","Cl","Br","I","H"]

def read_sdf(file_path,prop_name,conditions = None):
    print(f'Reading {file_path}')
    try:
        suppl = Chem.SDMolSupplier(file_path)
        data = []
        for mol in tqdm(suppl):
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                prop_val = float(mol.GetProp(f'{prop_name}')) if mol.HasProp(f'{prop_name}') else None
                data.append({'SMILES': smiles, prop_name: prop_val})
        return data
    except OSError:
        print(f'Error reading {file_path}')
        return None
