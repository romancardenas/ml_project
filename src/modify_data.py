import re
import pandas as pd
import numpy as np


def to_binary(data, columns):
    """Returns a copy of a dataframe with binarized columns.

        Keyword arguments:
        -- data: dataframe
        -- columns: dictionary with dataframe column names to be binarized and new column names for each one

        Returns:
        -- out_data: copy of original dataframe with binarized columns"""
    original_columns = list(columns.keys())

    out_data = data.copy()
    out_data[original_columns] = (data[original_columns] != 0).astype(int)
    out_data = out_data.rename(columns=columns)
    print(out_data)
    print(out_data.columns)

    return out_data


def one_to_K(data, columns):
    """Returns a copy of a dataframe with 1-out-of-K new columns.

            Keyword arguments:
            -- data: dataframe
            -- columns: dataframe column names to be modified to 1-out-of-K columns
            Returns:
            -- out_data: copy of original dataframe with 1-out-of-K columns
            -- new_columns: list with names of new generated columns"""
    out_data = data.copy()
    new_columns = list()
    for column in columns:
        data_dummies = pd.get_dummies(out_data[column], prefix=column)
        new_columns.extend(list(data_dummies))
        out_data = pd.concat([out_data, data_dummies], axis=1)  # add new generated columns
        out_data = out_data.drop(column, axis=1)  # remove original column
    return out_data, new_columns


def date_to_month(data, columns):
    """Returns a copy of a dataframe with date columns modified to months.

                Keyword arguments:
                -- data: dataframe
                -- columns: dataframe column names to be modified to months

                Returns:
                -- out_data: copy of original dataframe with months instead of dates"""
    original_columns = list(columns.keys())
    out_data = data.copy()
    for column in original_columns:
        c_list = out_data[column].tolist()
        new_column = [int(re.findall("^[0-9]{4}([0-9]{2})[0-9]{2}T[0-9]{6}$", a)[0]) for a in c_list]
        out_data[column] = new_column
        print(out_data[column])
    out_data = out_data.rename(columns=columns)
    return out_data


def normalize(data, omit_columns=None):
    """Returns a copy of a dataframe with normalized columns.

                Keyword arguments:
                -- data: dataframe
                -- omit_columns: dataframe column labels not to be normalized  (default None)

                Returns:
                -- out_data: copy of original dataframe with desired normalized columns
                -- out_data_mean: pandas series with the mean of each column
                -- out_data_std: pandas series with the standard deviation of each column

        NOTE: actually, it does not return the mean nor the standard deviation, but the parameters that has been applied
                For example, if a column has been omitted, the returned mean will be 0, and the standard deviation 1
                This aspect simplifies the data reconstruction process."""
    out_data = data.copy()
    out_data_mean = out_data.mean()
    out_data_std = out_data.std()
    if omit_columns is not None:
        for i in data.columns:
            if i in omit_columns:
                out_data_mean[i] = 0
                out_data_std[i] = 1
    out_data = (out_data - out_data_mean) / out_data_std
    return out_data, out_data_mean, out_data_std


def remove_outliers(data, target_columns=None, std=3):
    """Returns a copy of a dataframe without outliers.

                    Keyword arguments:
                    -- data: dataframe
                    -- target_columns: columns of the dataframe you want to remove the outliers from (default None)
                    -- std: maximum deviation allowed relative to the standard deviation  (default 3)

                    Returns:
                    -- out_data: copy of original dataframe without outliers"""
    out_data = data.copy()
    if target_columns is None:
        target_columns = list(out_data)
    out_data_mean = out_data.mean()
    out_data_std = out_data.std()
    for column in target_columns:
        out_data = out_data[np.abs(out_data[column] - out_data_mean[column]) <= std * out_data_std[column]]
    return out_data
