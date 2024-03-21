
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from serenityff.charge.utils import Atom, Bond, Molecule
import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from typing import Any, Optional, Union
from scipy.stats import mannwhitneyu

allowable_set= ["C","N","O","F","P","S","Cl","Br","I","H"]
default_properties = ['LogVP', 'LogP', 'LogOH', 'LogBCF', 'LogHalfLife', 'BP', 'Clint', 'FU', 'LogHL', 'LogKmHL', 'LogKOA', 'LogKOC', 'MP', 'LogMolar']

class CustomData(Data):
    """
    Data Class holding the pyorch geometric molecule graphs.
    Similar to pyg's data class but with two extra attributes,
    being smiles and molecule_charge.
    """

    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        smiles: str = None,
        molecule_charge: int = None,
    ):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            smiles=smiles,
            molecule_charge=molecule_charge,
        )

    @property
    def smiles(self) -> Union[str, None]:
        return self["smiles"] if "smiles" in self._store else None

    @property
    def molecule_charge(self) -> Union[int, None]:
        return self["molecule_charge"] if "molecule_charge" in self._store else None

    def __setattr__(self, key: str, value: Any):
        if key == "smiles":
            return self._set_smiles(value)
        elif key == "molecule_charge":
            return self._set_molecule_charge(value)
        else:
            return super().__setattr__(key, value)

    def _set_smiles(self, value: str) -> None:
        """
        Workaround for the ._store that is implemented in pytorch geometrics data.

        Args:
            value (str): smiles to be set

        Raises:
            TypeError: if value not convertable to string.
        """
        if isinstance(value, str) or value is None:
            return super().__setattr__("smiles", value)
        else:
            raise TypeError("Attribute smiles has to be of type string")

    def _set_molecule_charge(self, value: int) -> None:
        """
        Workaround for the ._store that is implemented in pytorch geometrics data.

        Args:
            value (int): molecule charge to be set.

        Raises:
            TypeError: if value not integer.

        """
        if isinstance(value, int):
            return super().__setattr__("molecule_charge", torch.tensor([value], dtype=int))
        elif value is None:
            return super().__setattr__("molecule_charge", None)
        elif isinstance(value, float) and value.is_integer():
            return super().__setattr__("molecule_charge", torch.tensor([int(value)], dtype=int))
        else:
            raise TypeError("Value for charge has to be an int!")


def get_graph_from_mol(
    mol: Molecule,
    index: int,
    allowable_set: Optional[list[str]] = [
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "H",
    ],
    no_y: Optional[bool] = False,
) -> CustomData:
    """
    Creates an pytorch_geometric Graph from an rdkit molecule.
    The graph contains following features:
        > Node Features:
            > Atom Type (as specified in allowable set)
            > formal_charge
            > hybridization
            > H acceptor_donor
            > aromaticity
            > degree
        > Edge Features:
            > Bond type
            > is in ring
            > is conjugated
            > stereo
    Args:
        mol (Molecule): rdkit molecule
        allowable_set (Optional[List[str]], optional): List of atoms to be \
            included in the feature vector. Defaults to \
                [ "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", ].

    Returns:
        CustomData: pytorch geometric Data with .smiles as an extra attribute.
    """
    grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if not no_y:
        graph.y = torch.tensor(
            [float(x) for x in mol.GetProp("MBIScharge").split("|")],
            dtype=torch.float,
        )
    else:
        graph.y = torch.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=torch.float,
        )
    #TODO: Check if batch is needed, otherwise this could lead to a problem if all batches are set to 0
    # Batch will be overwritten by the DataLoader class
    graph.batch = torch.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph

def read_sdf(file_path,prop_name,conditions = None):
    """
    Reads a Structure-Data File (SDF) and extracts the SMILES representation and a specified property for each molecule.

    Parameters:
    file_path (str): The path to the SDF file.
    prop_name (str): The name of the property to extract from each molecule.
    conditions (dict, optional): A dictionary of conditions to apply when reading the file. Not implemented in this function.

    Returns:
    list: A list of dictionaries, where each dictionary contains the SMILES representation and the specified property of a molecule.
    If an error occurs while reading the file, the function returns None.

    """
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
    

def load_data(overwrite=False,prop_list = default_properties):
    """
    Loads data from a CSV file or reads from SDF files if the CSV file is not found or if overwrite is True.

    Parameters:
    overwrite (bool): If True, the function will ignore the CSV file and read from the SDF files. Default is False.
    prop_list (list): A list of properties to extract from the SDF files. Default is the list of default properties.

    Returns:
    DataFrame: A pandas DataFrame containing the SMILES representation and the specified properties of the molecules.

    The DataFrame is loaded from a CSV file if it exists and overwrite is False. If the CSV file does not exist or if overwrite is True, the function reads the SDF files, extracts the SMILES representation and the specified properties for each molecule, removes duplicates based on the SMILES representation, and merges the data into a single DataFrame.
    """
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


def plot_property_histograms(df):
    """
    Plots histograms of the properties in the given DataFrame.

    Parameters:
    df (DataFrame): A pandas DataFrame where each column represents a property. The first column is ignored.

    The function creates a 3x5 grid of subplots and plots a histogram for each property in the DataFrame. The title of each subplot is the name of the property.
    """
    fig, axes = plt.subplots(2, 7, figsize=(20, 5.5))
    for i, (col, ax) in enumerate(zip(df.columns[1:], axes.flatten())):
        sns.histplot(df[col], ax=ax, kde=True, bins=50, color='skyblue', edgecolor='black')
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    plt.show()

def scale_props(df,prop_list = default_properties):
    """
    Scales the properties in the given DataFrame using MinMaxScaler.

    Parameters:
    df (DataFrame): A pandas DataFrame where each column represents a property.
    prop_list (list): A list of properties to scale. Default is the list of default properties.

    Returns:
    DataFrame, MinMaxScaler: A new DataFrame where the specified properties have been scaled to the range [0, 1], and the fitted MinMaxScaler object.

    The function creates a copy of the input DataFrame, scales the specified properties using MinMaxScaler, and replaces the original properties with the scaled properties in the copied DataFrame.
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[prop_list] = scaler.fit_transform(df_scaled[prop_list])
    return df_scaled,scaler

def compare_distributions(df_scaled, train, test, figsize=(20,5.5)):
    """
    Compares the distributions of properties in the train and test datasets.

    Parameters:
    df_scaled (DataFrame): A pandas DataFrame where each column represents a property. The first column is ignored.
    train (DataFrame): The training dataset.
    test (DataFrame): The testing dataset.
    figsize (tuple): The size of the figure to create. Default is (15, 15).

    The function checks if the distribution of each property is similar in the train and test datasets using the Mann-Whitney U test. It then creates a grid of subplots and plots a boxplot for each property in the train and test datasets. The title of each subplot is the name of the property. The y label is 'Normalized value'.
    """
    # Check if for each property the distribution is similar in train and test
    for col in df_scaled.columns[1:]:
        train_tmp = train[col].dropna()
        test_tmp = test[col].dropna()
        print(f'{col}: {mannwhitneyu(train_tmp,test_tmp)}')

    num_cols = len(df_scaled.columns[1:])
    num_rows = num_cols // 7 if num_cols % 7 == 0 else num_cols // 7 + 1
    fig, axes = plt.subplots(num_rows, 7, figsize=figsize)

    for i, col in enumerate(df_scaled.columns[1:]):
        train_tmp = train[col].dropna()
        test_tmp = test[col].dropna()
        row = i // 7
        col_i = i % 7
        ax = axes[row, col_i]
        ax.boxplot(x=[train_tmp,test_tmp],labels=['train','test'])
        ax.set_xticklabels(['train', 'test'])
        ax.set_ylabel('Normalized value')
        ax.set_title(col)
    plt.tight_layout()
    plt.show()
    
    
def load_graphs(dash_charges=False, scaled=True):
    """
    Load molecular graphs from saved files.

    Parameters:
    - dash_charges (bool): Whether to load graphs with dash charges. Default is False.
    - scaled (bool): Whether to load scaled graphs. Default is True.

    Returns:
    - mol_graphs: Loaded molecular graphs.
    """
    if dash_charges:
        if scaled:
            mol_graphs = torch.load('mol_graphs_dash_charges_scaled_train.pt')
        else:
            mol_graphs = torch.load('mol_graphs_dash_charges_unscaled_train.pt')
    else:
        if scaled:
            mol_graphs = torch.load('mol_graphs_unscaled_train.pt')
        else:
            mol_graphs = torch.load('mol_graphs_scaled_train.pt')
    return mol_graphs


def save_graphs_func(mol_graphs, dash_charges=False, scaled=True):
    """
    Save the molecular graphs to a file.

    Parameters:
        mol_graphs (torch.Tensor): The molecular graphs to be saved.
        dash_charges (bool, optional): Whether to include dash charges in the file name. Defaults to False.
        scaled (bool, optional): Whether the graphs are scaled. Defaults to True.
    """
    if dash_charges:
        if scaled:
            torch.save(mol_graphs, 'mol_graphs_dash_charges_scaled_train.pt')
        else:
            torch.save(mol_graphs, 'mol_graphs_dash_charges_unscaled_train.pt')
    else:
        if scaled:
            torch.save(mol_graphs, 'mol_graphs_unscaled_train.pt')
        else:
            torch.save(mol_graphs, 'mol_graphs_scaled_train.pt')


def get_graphs(df,dash_charges=False,scaled =True,save_graphs = False):
    """
    Get molecular graphs from the given DataFrame.
    
    Parameters:
    df (DataFrame): A pandas DataFrame containing the SMILES representation and the specified properties of the molecules.
    dash_charges (bool, optional): Whether to use dash charges. Defaults to False.
    scaled (bool, optional): Whether to scale the graphs. Defaults to True.
    save_graphs (bool, optional): Whether to save the graphs to a file. Defaults to False.
    
    Returns:
    mol_graphs: The molecular graphs.
    """
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
        from serenityff.charge.tree.dash_tree import DASHTree
        tree = DASHTree(tree_folder_path='props2') #add path to tree_folder_path if needed ex. tree_folder_path='/localhome/..../props2'
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
        #get the smiles of indices_to_drop_total
        smiles_to_drop = [df['SMILES'].iloc[i] for i in indices_to_drop_total]
        if indices_to_drop_total:
            #drop these smiles from the dataframe
            df = df[~df['SMILES'].isin(smiles_to_drop)]
    
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