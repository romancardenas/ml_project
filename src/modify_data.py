def to_binary(data, columns):
    """Returns a copy of a dataframe with binarized columns.

        Keyword arguments:
        -- data: dataframe
        -- columns: dictionary with dataframe column names to be binarized and new column names for each one

        Returns:
        -- out_data: copy of original dataframe with binarized columns"""
    original_columns = list(columns.keys())
    new_columns = list(columns.values())
    #original_columns = [item[0] for item in columns]
    #new_columns = [item[1] for item in columns]

    out_data = data.copy()
    out_data[original_columns] = (data[original_columns] != 0).astype(int)
    out_data = out_data.rename(columns=columns)
    #out_data = out_data.drop(original_columns, 1)
    print(out_data)
    print(out_data.columns)

    return out_data


def one_to_K(data, columns):  # TODO
    """Returns a copy of a dataframe with 1-out-of-K new columns.

            Keyword arguments:
            -- data: dataframe
            -- columns: dataframe column names to be modified to 1-out-of-K columns

            Returns:
            -- out_data: copy of original dataframe with 1-out-of-K columns"""
    return data.copy()


def date_to_month(data, columns):  # TODO
    """Returns a copy of a dataframe with date columns modified to months.

                Keyword arguments:
                -- data: dataframe
                -- columns: dataframe column names to be modified to months

                Returns:
                -- out_data: copy of original dataframe with months instead of dates"""
    return data.copy()


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
    out_data_std = out_data.max() - out_data.min()
    if omit_columns is not None:
        for i in data.columns:
            if i in omit_columns:
                out_data_mean[i] = 0
                out_data_std[i] = 1
    out_data = (out_data - out_data_mean) / out_data_std
    return out_data, out_data_mean, out_data_std
