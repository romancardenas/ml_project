from src.read_data import *
from src.modify_data import *


data_path = '../data/kc_house_data.csv'
options = {
    'omit_columns': ['id', 'date', 'sqft_living', 'view', 'sqft_lot', 'grade', 'lat', 'long'],
    'binary_columns': {'sqft_basement': 'basement'},
    'one_to_k': [],
    'date_to_month': [],
    'no_normalized_columns': ['price'],
    'train_size': 0.75,
    'test_size': 0.25
}

# Get data from file
data = read_data(data_path, omit_columns=options['omit_columns'])

# Transform binary columns
data = to_binary(data, options['binary_columns'])

# Transform 1-to-K columns
data = one_to_K(data, options['one_to_k'])

# Transform date columns to month
data = date_to_month(data, options['date_to_month'])

# Normalize columns  # TODO Maybe the normalization must be done only with the train set
data, data_mean, data_std = normalize(data, omit_columns=options['no_normalized_columns'])

# Divide data between train and test
data_train, data_test = divide_data(data, options['train_size'])

attributeNames = list(data.columns.values)
y = data_train.values[:, :1]
X = data_train.values[:, 1:]
