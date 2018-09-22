import pandas as pd


def read_data(data_path, omit_columns=None):
    """Returns data from csv file.

    Keyword arguments:
    -- data_path: path to the csv file
    -- omit_columns: list of columns you want to drop from dataset (default None)

    Returns:
    -- data: data from csv file (dataframe)"""

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
    data_train = data.loc[:threshold]
    data_test = data.loc[threshold:]
    return data_train, data_test
