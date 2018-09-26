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
                     'sqft_lot15',
                     'zipcode'],
    'binary_columns': {},
    'date_to_month': {},
    'one_to_k': [],  # The month, once extracted from date, will also turn to 1-out-of-K column
    'no_normalized_columns': ['price'],
    'train_size': 0.75,
}

# Get data from file
data = read_data(data_path, omit_columns=options['omit_columns'])

a = sns.xkcd_palette(['green', 'pinkish purple', 'blue', 'purplish', 'grape purple', 'deep purple'])
sns.set_palette(a)

# Print all the attributes against each other
print(list(data))
#for i in list(data):
#    plt.figure()
#    plt.scatter(data[i], data.price, marker=".",)
#    plt.title("{} against price".format(i))
#    plt.ylabel("Price (USD)")
#    plt.xlabel("{}".format(i))
#    sns.despine()

# plt.figure()
#
# plt.scatter(data.yr_built, data.price, marker=".", c = 'black')
# #sns.regplot(data.floors, data.price, order=1, scatter=False, label='regression line')
# plt.title("Year Built against Price", fontsize = 25)
# plt.ylabel("Price (USD)", fontsize = 15)
# plt.xlabel("Year Built", fontsize = 15)
# plt.scatter(data.yr_built, data.price, marker=".", c = 'black')
# sns.despine()

# Transform binary columns
data = to_binary(data, options['binary_columns'])

# Transform date columns to month
data = date_to_month(data, options['date_to_month'])

# Normalize columns
# TODO If normalization has to be done to train data, first we have to binarize and do the 1-out-of-K to all the data set
# TODO Then, divide in train and test
# TODO then normalize ONLY train data set, quiting the binaries
data, data_mean, data_std = normalize(data, omit_columns=options['no_normalized_columns'])

# Transform 1-to-K columns
data = one_to_K(data, options['one_to_k'])

# Divide data between train and test
data_train, data_test = divide_data(data, options['train_size'])


# Convert pandas dataframe to NumPy ndarray (compatibility with examples code)
attributeNames = list(data.columns.values)
attributeNames.remove('price')
y = data_train.values[:, 0]
X = data_train.values[:, 1:]
