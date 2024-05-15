from rdkit import Chem
from tqdm import tqdm
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import mannwhitneyu

default_properties = ['LogVP', 'LogP', 'LogOH', 'LogBCF', 'LogHalfLife', 'BP', 'Clint', 'FU', 'LogHL', 'LogKmHL', 'LogKOA', 'LogKOC', 'MP', 'LogMolar']


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
