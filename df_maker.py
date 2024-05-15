import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, default='property')
args = parser.parse_args()


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

train_prop = train[['SMILES', args.property]]
train_prop = train_prop.dropna()
train_prop.reset_index(drop=True, inplace=True)
train_prop.to_csv(f'train_{args.property}.csv', index=False)

test_prop = test[['SMILES', args.property]]
test_prop = test_prop.dropna()
test_prop.reset_index(drop=True, inplace=True)
test_prop.to_csv(f'test_{args.property}.csv', index=False)
