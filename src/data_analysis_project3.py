import matplotlib.pyplot as plt
from src.modify_data import *
from scipy.stats import zscore

data_path = '../data/SAheart.csv'

# Get data from file
data = pd.read_csv(data_path)
data = data.drop(columns=['row.names'])
aux = pd.get_dummies(data['famhist'], prefix=None)
data[['famhist']] = aux[['Present']]
del aux

# boxplot of original non-binary data to see outliers
plt.figure(1)
plt.title('South African Heart Disease: Boxplot (original)')
data.boxplot()
print(len(list(data)))
plt.xticks(range(1, len(list(data)) + 1), list(data), rotation=45)
plt.show()

# boxplot of original standardized non-binary data to see more clearly outliers
plt.figure(2)
plt.title('South African Heart Disease: Boxplot (original standarized)')
plt.boxplot(zscore(data, ddof=1))
plt.xticks(range(1, len(list(data)) + 1), list(data), rotation=45)
plt.show()

# We will remove outliers ONLY if the column in question is not binary
std = 1.5
data_mean = data.mean()
data_std = data.std()
for column in list(data):
    data = data[np.abs(data[column] - data_mean[column]) <= std * data_std[column]]

# boxplot of regularized non-binary data
plt.figure(3)
plt.title('South African Heart Disease: Boxplot (without outliers)')
data.boxplot()
plt.xticks(range(1, len(list(data)) + 1), list(data), rotation=45)
plt.show()

# boxplot of regular standardized data
plt.figure(4)
plt.title('South African Heart Disease: Boxplot (without outliers and standarized)')
plt.boxplot(zscore(data, ddof=1), list(data))
plt.xticks(range(1, len(list(data)) + 1), list(data), rotation=45)
plt.show()

aux = data[['chd']]
data = data.drop('chd', axis=1)
data.insert(loc=0, column='chd', value=aux)
del aux
data.to_csv('../data/SAheart_reg.csv', index=False)
