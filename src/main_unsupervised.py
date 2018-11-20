import matplotlib.pyplot as plt
from src.read_data import *
from src.modify_data import *
from scipy.stats import zscore


data_path = '../data/kc_house_data.csv'
options = {
    'omit_columns': ['id',
                     'date',
                     'bedrooms',
                     'bathrooms',
                     'floors',
                     'waterfront',
                     'view',
                     'condition',
                     'grade',
                     'sqft_above',
                     'sqft_basement',
                     'yr_renovated',
                     'sqft_living15',
                     'sqft_lot15'],
    'binary_columns': {},
    'date_to_month': {},
    'one_to_k': ['zipcode'],  # The month, once extracted from date, will also turn to 1-out-of-K column
    'non_normalized_columns': ['price']
}

# Get data from file
#data = read_data(data_path)
data = read_data(data_path, omit_columns=options['omit_columns'])
# Transform 1-to-K columns
data, new_columns_K = one_to_K(data, options['one_to_k'])

plt.figure(figsize=(12, 9))
for zipcode in new_columns_K:
    aux = data.loc[data[zipcode] == 1]
    plt.plot(aux['long'], aux['lat'], '.')
plt.show()

# Transform binary columns
data = to_binary(data, options['binary_columns'])

# Transform date columns to months (if apply)
data = date_to_month(data, options['date_to_month'])

# Transform 1-to-K columns
#data, new_columns_K = one_to_K(data, options['one_to_k'])

# removing outliers
new_columns_K.extend(options['binary_columns'].values())  # Add binary columns to one-to-K columns
outliers_to_remove = [column for column in list(data) if column not in new_columns_K and column != 'lat' and column != 'long']  # list of target columns

# We will plot boxplots of all the data that is not binary (that is, its name is not in new_columns_K list)
labels_aux = list()
for column in list(data):
    if column not in new_columns_K:
        labels_aux.append(column)
# Numpy array with non-binary data
data_aux = data[labels_aux].values

# boxplot of original non-binary data to see outliers
plt.figure()
plt.title('House prizing: Boxplot (original)')
plt.boxplot(data_aux)
plt.xticks(range(1, len(labels_aux) + 1), labels_aux, rotation=45)
plt.show()

# boxplot of original standardized non-binary data to see more clearly outliers
plt.figure()
plt.title('House prizing: Boxplot (original standarized)')
plt.boxplot(zscore(data_aux, ddof=1), labels_aux)
plt.xticks(range(1, len(labels_aux) + 1), labels_aux, rotation=45)
plt.show()

# It is clear that our dataset is full of outliers. We will follow the next outlier removal policy:
# - There are houses extremely expensive, that they are not expensive because of their size or other objective reason,
#   but maybe for other reasons (materials, history, etc...) that is not reflected in our dataset. If we keep this data,
#   our model will decrease its performance.
# - Other features, such as number of rooms or bathrooms, have clear outliers that must be removed.
# - the Square feet of the lot feature has a huge amount of outliers. However, this is somehow reasonable: it may be
#   houses with big gardens etc that increase a lot the total size of the house.
#   This feature has also a big variance: there are flats without garden, others with huge ones...
# We have decided to remove the outliers according to this policy:
# - If the feature is greater or bigger than 1.5 times the standard deviation of the data, we will remove it.
data = remove_outliers(data, target_columns=outliers_to_remove, std=1.5)  # remove outliers
data_aux = data[labels_aux].values

# boxplot of original non-binary data to see outliers
plt.figure()
plt.title('House prizing: Boxplot (without outliers)')
plt.boxplot(data_aux)
plt.xticks(range(1, len(labels_aux) + 1), labels_aux, rotation=45)
plt.show()

# boxplot of original standardized non-binary data to see more clearly outliers
plt.figure()
plt.title('House prizing: Boxplot (without outliers and standarized)')
plt.boxplot(zscore(data_aux, ddof=1), labels_aux)
plt.xticks(range(1, len(labels_aux) + 1), labels_aux, rotation=45)
plt.show()

plt.figure(figsize=(12, 9))
for zipcode in new_columns_K:
    aux = data.loc[data[zipcode] == 1]
    plt.plot(aux['long'], aux['lat'], '.')
plt.show()

data.to_csv('../data/kc_house_data_project_3.csv', index=False)
