import matplotlib.pyplot as plt
from src.read_data import *
from src.modify_data import *
import seaborn as sns


data_path = '../data/kc_house_data.csv'
options = {
    'omit_columns': ['id',
                     'view',
                     'lat',
                     'date',
                     'long',
                     'yr_renovated',
                     'waterfront',
                     'yr_built',
                     'grade',
                     'sqft_living15',
                     'sqft_lot15'],
    'binary_columns': {},
    'date_to_month': {},
    'one_to_k': ['zipcode'],  # The month, once extracted from date, will also turn to 1-out-of-K column
    'non_normalized_columns': ['price'],
    'train_size': 0.75,
}

# TODO think about a more flexible script for indicating which cross-validation method we want to use

# Plotting style stuff
a = sns.xkcd_palette(['green', 'pinkish purple', 'blue', 'purplish', 'grape purple', 'deep purple'])
sns.set_palette(a)

# Get data from file
data = read_data(data_path, omit_columns=options['omit_columns'])

# Transform binary columns
data = to_binary(data, options['binary_columns'])

# Transform date columns to months
data = date_to_month(data, options['date_to_month'])

# Transform 1-to-K columns
data, new_columns_K = one_to_K(data, options['one_to_k'])

# Normalize columns
non_normalized = options['non_normalized_columns']
non_normalized.extend(options['binary_columns'].values())  # Add binary columns to non-normalized columns
non_normalized.extend(new_columns_K)  # Add one-out-of-K columns to non-normalized columns
data, data_mean, data_std = normalize(data, omit_columns=non_normalized)

# TODO removing outliers
# TODO storing clean data to CSV files to avoid rerunning

# Divide data between train and test
# TODO this only accepts holdout cross-validation method, shoud we think about K-fold?
# TODO If normalization has to be done to train data, first we have to divide in train and test
# TODO then normalize ONLY train data set
data_train, data_test = divide_data(data, options['train_size'])

# Convert pandas dataframe to NumPy ndarray (compatibility with examples code)
# TODO this was useful for project 1, maybe we have to change this for the second project
attributeNames = list(data.columns.values)
attributeNames.remove('price')
y = data_train.values[:, 0]
X = data_train.values[:, 1:]
