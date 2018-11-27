import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.linalg import svd

data_path = '../data/Seed_Data.csv'

# Get data from file
data = pd.read_csv(data_path)
aux = data[['target']]
data = data.drop('target', axis=1)
data.insert(loc=0, column='target', value=aux)
del aux

# boxplot of original non-binary data to see outliers
plt.figure(1)
plt.title('Seeds: Boxplot (original)')
data.boxplot()
plt.xticks(range(1, len(list(data)) + 1), list(data), rotation=45)
plt.show()

# boxplot of original standardized non-binary data to see more clearly outliers
plt.figure(2)
plt.title('Seeds: Boxplot (original standarized)')
plt.boxplot(zscore(data, ddof=1))
plt.xticks(range(1, len(list(data)) + 1), list(data), rotation=45)
plt.show()

# We will remove outliers ONLY if the column in question is not binary
std = 2
data_mean = data.mean()
data_std = data.std()
for column in list(data):
    data = data[np.abs(data[column] - data_mean[column]) <= std * data_std[column]]

# boxplot of regularized non-binary data
plt.figure(3)
plt.title('Seeds: Boxplot (without outliers)')
data.boxplot()
plt.xticks(range(1, len(list(data)) + 1), list(data), rotation=45)
plt.show()

# boxplot of regular standardized data
plt.figure(4)
plt.title('Seeds: Boxplot (without outliers and standarized)')
plt.boxplot(zscore(data, ddof=1), list(data))
plt.xticks(range(1, len(list(data)) + 1), list(data), rotation=45)
plt.show()


data.to_csv('../data/seed_data_reg.csv', index=False)


# PCA ANALYSIS
attributeNames = list(data.columns.values)
attributeNames.remove('target')
y = data.values[:, 0]
X = data.values[:, 1:]
M, N = X.shape

# Standardize data
#Y = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (X - X.mean(axis=0))

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)
V = V.T

# Compute variance explained by principal components
rho = (S * S) / (S * S).sum()
cum_rho = rho.cumsum()
pca_index_90 = 0
for i in range(len(cum_rho)):
    if cum_rho[i] > 0.9:
        pca_index_90 = i + 1
        break
print('With {0:d} PCAs you have {1:.2f}% of the information'.format(pca_index_90, cum_rho[pca_index_90] * 100))

# Plot variance explained
plt.figure(5)
plt.plot(range(1, len(rho) + 1), rho, 'o-')
plt.title('Variance Explained by Principal Components')
plt.xlabel('Principal components')
plt.ylabel('Variance explained')

plt.figure(6)
plt.plot(range(1, len(cum_rho) + 1), cum_rho, 'o-')
plt.axhline(y=0.9, color='r', linestyle='--')
plt.title('Cumulative Variance Explained by Principal Components')
plt.xlabel('Principal component')
plt.ylabel('Cumulative variance explained')
plt.show()

# Project data onto principal component space
Z = Y @ V
plt.figure(7)
plt.title("Wheat Seed Types Represented by the First two PCs")
plt.plot(Z[y == 0, 0], Z[y == 0, 1], '.')
plt.plot(Z[y == 1, 0], Z[y == 1, 1], '.')
plt.plot(Z[y == 2, 0], Z[y == 2, 1], '.')
plt.legend(['Type I', 'Type II', 'Type III'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
