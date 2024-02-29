
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch


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
    
def load_data(overwrite=False):

    if not overwrite:
        try:
            # Load data from file
            print('Loading data from file')
            df = pd.read_csv('OPERA_Data/OPERA_properties.csv')
            return df
        except FileNotFoundError:
            print('File not found, reading from SDF files')

    file_paths = {
        'LogVP': 'OPERA_Data/VP_QR.sdf',
        'LogP': 'OPERA_Data/LogP_QR.sdf',
        'LogOH': 'OPERA_Data/AOH_QR.sdf',
        'LogBCF': 'OPERA_Data/BCF_QR.sdf',
        'LogHalfLife': 'OPERA_Data/Biodeg_QR.sdf',
        'BP': 'OPERA_Data/BP_QR.sdf',
        'Clint': 'OPERA_Data/Clint_QR.sdf',
        'FU': 'OPERA_Data/FU_QR.sdf',
        'LogHL': 'OPERA_Data/HL_QR.sdf',
        'LogKmHL': 'OPERA_Data/KM_QR.sdf',
        'LogKOA': 'OPERA_Data/KOA_QR.sdf',
        'LogKOC': 'OPERA_Data/KOC_QR.sdf',
        'MP': 'OPERA_Data/MP_QR.sdf',
        'LogMolar': 'OPERA_Data/WS_QR.sdf',
    }

    # Read SDF files and extract data
    data_dict = {prop: read_sdf(path, prop) for prop, path in file_paths.items()}

    # Create Pandas dataframes
    df_dict = {prop: pd.DataFrame(data) for prop, data in data_dict.items()}

    #for eaach df in df_dict, remove duplicates
    for df in df_dict.values():
        df.drop_duplicates(subset='SMILES', inplace=True)

    df_combined = reduce(lambda left, right: pd.merge(left, right, on='SMILES', how='outer'), [df[['SMILES', prop]] for prop, df in df_dict.items()])
    df_combined = df_combined[['SMILES'] + list(df_dict.keys())]
    return df_combined


def scale_props(df):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[['LogVP', 'LogP', 'LogOH', 'LogBCF', 'LogHalfLife', 'BP', 'Clint', 'FU', 'LogHL', 'LogKmHL', 'LogKOA', 'LogKOC', 'MP', 'LogMolar']] = scaler.fit_transform(df_scaled[['LogVP', 'LogP', 'LogOH', 'LogBCF', 'LogHalfLife', 'BP', 'Clint', 'FU', 'LogHL', 'LogKmHL', 'LogKOA', 'LogKOC', 'MP', 'LogMolar']])
    return df_scaled,scaler

def load_graphs(dash_charges=False,scaled =True):
    if dash_charges:
        if scaled:
            mol_graphs = torch.load('mol_graphs_dash_charges_scaled.pt')
        else:
            mol_graphs = torch.load('mol_graphs_dash_charges_unscaled.pt')
    else:
        if scaled:
            mol_graphs = torch.load('mol_graphs_unscaled.pt')
        else:
            mol_graphs = torch.load('mol_graphs_scaled.pt')
    return mol_graphs

def save_graphs_func(mol_graphs,dash_charges=False,scaled =True):
    if dash_charges:
        if scaled:
            torch.save(mol_graphs, 'mol_graphs_dash_charges_scaled.pt')
        else:
            torch.save(mol_graphs, 'mol_graphs_dash_charges_unscaled.pt')
    else:
        if scaled:
            torch.save(mol_graphs, 'mol_graphs_unscaled.pt')
        else:
            torch.save(mol_graphs, 'mol_graphs_scaled.pt')

def get_graphs(df,dash_charges=False,scaled =True,save_graphs = False):

    try:
        mol_graphs = load_graphs(dash_charges,scaled)
        print('Loading previously created graphs')
        return mol_graphs
    except FileNotFoundError:
        print('Creating new graphs')

    all_mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
    mols = [m for m in all_mols if m.GetNumAtoms() > 1]
    error_mol = [m for m in all_mols if m.GetNumAtoms() <= 1]
    indices_to_drop_size = [all_mols.index(m) for m in error_mol]
     
    if dash_charges:
        from custom_featurization_stuff import get_graph_from_mol
        from serenityff.charge.tree.dash_tree import DASHTree
        tree = DASHTree(tree_folder_path='/localhome/cschiebroek/other/serenityff-charge/tree')
        tmp_mols = mols 
        mols = []
        error_mols_charges = []
        for m in tqdm(tmp_mols):
            try:
                mol = Chem.AddHs(m, addCoords=True)
                charges = tree.get_molecules_partial_charges(mol,chg_std_key='std',chg_key='result')["charges"]
            except:
                error_mols_charges.append(m)
                continue
            for i,atom in enumerate(mol.GetAtoms()):
                atom.SetDoubleProp('charge',charges[i])
            mols.append(mol)
        indices_to_drop_charges = [all_mols.index(m) for m in error_mols_charges]
        indices_to_drop_total = list(set(indices_to_drop_size + indices_to_drop_charges))
        print(len(indices_to_drop_total), len(indices_to_drop_size), len(indices_to_drop_charges))
        if indices_to_drop_total:
            print('Caution! Mols dropped')
            df = df.drop(indices_to_drop_total)
    
    else:
        from serenityff.charge.gnn.utils import get_graph_from_mol
        if indices_to_drop_size:
            print(f'Caution! {len(indices_to_drop_size)} Mols dropped')
            df = df.drop(indices_to_drop_size)

    ys = df.iloc[:, 1:].values
    ys = np.nan_to_num(ys, nan=-1)
    y = torch.tensor(ys, dtype=torch.float32)
    y = y.unsqueeze(1)
    assert len(mols) == len(y)
    mol_graphs = [get_graph_from_mol(mol,i, allowable_set,no_y=True) for i,mol in enumerate(mols)]
    SMILES = df['SMILES'].tolist()
    assert len(mol_graphs) == len(y) == len(SMILES)
    for i in range(len(mol_graphs)):
        mol_graphs[i].SMILES = SMILES[i]
        mol_graphs[i].y = y[i]
    if save_graphs:
        save_graphs_func(mol_graphs,dash_charges,scaled)
    return mol_graphs