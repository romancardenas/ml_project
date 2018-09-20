import pandas as pd
from matplotlib.pyplot import *
from src.modify_data import *


def read_data(data_path, omit_columns=None):
    """Returns data from csv file.

    Keyword arguments:
    -- data_path: path to the csv file
    -- omit_columns: list of columns you want to drop from dataset (default None)

    Returns:
    -- attributeNames: List with the name of each attribute
    -- X: houses data
    -- y: houses price"""

    data = pd.read_csv(data_path)
    if omit_columns:
        data = data.drop(columns=omit_columns)

    # Extract class names to python list, then encode with integers (dict)
    attributeNames = list(data.columns.values)
    data = data.values
    X = data[:, 1:]
    y = data[:, 0]
    return X, y, attributeNames


def divide_data(data, train_size=0.75):
    """Divides dataframe for training and testing.

        Keyword arguments:
        -- data: original dataframe
        -- train_size: percentage of observations you want in the training dataset (dafault 0.75)"""
    N = data.shape[0]
    threshold = int(N * train_size)
    data_train = data[:threshold]
    data_test = data[threshold:]
    return data_train, data_test


if __name__ == '__main__':
    data_path = '../data/kc_house_data.csv'
    options = {
        'omit_columns': ['id', 'date', 'sqft_living', 'view', 'sqft_lot', 'grade', 'lat', 'long'],
        'binary_columns': ['sqft_basement'],
        'one_to_k': [],
        'train_size': 0.75,
        'test_size': 0.25
    }

    X, y, attributeNames = read_data(data_path, omit_columns=options['omit_columns'])

    binary = [attributeNames.index(x) - 1 for x in options['binary_columns'] if x in attributeNames]
    X = to_binary(X, binary)

    X, X_mean, X_std = normalize(X)
    print(X.mean(axis=0))
    print(X.std(axis=0))

    X_train, X_test = divide_data(X, options['train_size'])
    y_train, y_test = divide_data(y, options['train_size'])



