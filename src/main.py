from src.read_data import *
from src.modify_data import *


data_path = '../data/kc_house_data.csv'
options = {
    'omit_columns': ['id',
                     'sqft_living',
                     'view',
                     'sqft_lot',
                     'grade',
                     'lat',
                     'long',
                     'sqft_above'],
    'binary_columns': {'sqft_basement': 'basement',
                       'yr_renovated': 'renovated'},
    'date_to_month': {'date': 'month'},
    'one_to_k': ['zipcode',
                 'month'],  # The month, once extracted from date, will also turn to 1-out-of-K column
    'no_normalized_columns': ['price', 'zipcode', 'month', 'basement', 'renovated'],
    'train_size': 0.75,
}

# Get data from file
data = read_data(data_path, omit_columns=options['omit_columns'])

# Transform binary columns
data = to_binary(data, options['binary_columns'])

# Transform date columns to month
data = date_to_month(data, options['date_to_month'])

data, data_mean, data_std = normalize(data, omit_columns=options['no_normalized_columns'])


# Transform 1-to-K columns
data = one_to_K(data, options['one_to_k'], options['one_to_k'])

# Normalize columns
# TODO Maybe the normalization must be done only with the train set
# TODO Do we have to normalize also binaries?
#data, data_mean, data_std = normalize(data, omit_columns=options['no_normalized_columns', 'binary_columns', 'one_to_k'])

# Divide data between train and test
data_train, data_test = divide_data(data, options['train_size'])

# Convert pandas dataframe to NumPy ndarray (compatibility with examples code)
attributeNames = list(data.columns.values)
y = data_train.values[:, :1]
X = data_train.values[:, 1:]
