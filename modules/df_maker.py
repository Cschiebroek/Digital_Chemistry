import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils_data import load_data, compare_distributions

def create_init_data(seed=42, test_size=0.2):
    """
    Function to create initial data by splitting the combined dataset
    into training and test sets, and saving them to CSV files.
    """
    df_combined = load_data()
    train, test = train_test_split(df_combined, test_size=test_size, random_state=seed)
    
    # Print the distribution of each column in the test set
    for col in df_combined.columns[1:]:
        print(f'{col}: {test[col].count() / df_combined[col].count()}')
    
    compare_distributions(df_combined, train, test)
    print(f'Duplicated SMILES: {df_combined.SMILES.duplicated().sum()}')

    # Ensure the data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save the train and test datasets
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    
    print(f'Number of training datapoints: {len(train)}')
    print(f'Number of test datapoints: {len(test)}')

def main():
    parser = argparse.ArgumentParser(description='Process and prepare datasets.')
    parser.add_argument('--property', type=str, default='property', help='Property to process')
    parser.add_argument('--base_dir', type=str, default='', help='Base directory for data storage')
    args = parser.parse_args()

    # Ensure the base data directory exists
    base_data_dir = os.path.join(args.base_dir, 'data')
    if not os.path.exists(base_data_dir):
        os.makedirs(base_data_dir)
        create_init_data()

    # Load the datasets
    train_path = os.path.join(base_data_dir, 'train.csv')
    test_path = os.path.join(base_data_dir, 'test.csv')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Process and save the property-specific data
    train_prop = train[['SMILES', args.property]].dropna().reset_index(drop=True)
    train_prop.to_csv(os.path.join(base_data_dir, f'train_{args.property}.csv'), index=False)
    
    test_prop = test[['SMILES', args.property]].dropna().reset_index(drop=True)
    test_prop.to_csv(os.path.join(base_data_dir, f'test_{args.property}.csv'), index=False)

if __name__ == '__main__':
    main()
