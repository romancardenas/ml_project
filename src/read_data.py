import numpy as np
import pandas as pd
from matplotlib.pyplot import *
from src.modify_data import *



def read_data(data_path, omit_columns=None):
    """Returns data from csv file.

    Keyword arguments:
    -- data_path: path to the csv file
    -- omit_columns: list of columns you want to drop from dataset (default None)"""
    data = pd.read_csv(data_path)
    if omit_columns:
        data = data.drop(columns=omit_columns)
    return data


def divide_data(data, train_size=0.75):
    """Divides dataframe for training and testing.

        Keyword arguments:
        -- data: original dataframe
        -- train_size: percentage of observations you want in the training dataset (dafault 0.75)"""
    N, M = data.shape
    threshold = int(N * train_size)
    data_train = data.iloc[0:threshold]
    data_test = data.iloc[threshold:]
    return data_train, data_test


if __name__ == '__main__':
    data_path = '../data/kc_house_data.csv'
    options = {
        'omit_columns': ['id', 'date', 'sqft_living', 'view', 'sqft_lot', 'grade', 'lat', 'long'],
        'train_size': 0.75,
        'test_size': 0.25
    }
    data = read_data(data_path, omit_columns=options['omit_columns'])
    data_train, data_test = divide_data(data)
    print(data['sqft_basement'])
    print(to_binary(data, ['sqft_basement'])['sqft_basement'])
    print(data['sqft_basement'])


    print(data.describe())
    print(data.columns)
    print(data.size)
    print(data_train.size)
    print(data_test.size)
