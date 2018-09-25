# exercise 2.1.3
# (requires data structures from ex. 2.2.1)
from src.main import *

from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show
from scipy.linalg import svd

N = len(y)

# Subtract mean value from data
# Y = X - np.ones((N, 1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(X, full_matrices=False)
V = V.T

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()
cum_rho = rho.cumsum()
pca_index_90 = 0
for i in range(len(cum_rho)):
    if cum_rho[i] > 0.9:
        pca_index_90 = i + 1
        break
print('With {0:d} PCAs you have {1:.2f}% of the information'.format(pca_index_90, cum_rho[pca_index_90]*100))

# Plot variance explained
fig = figure()
ax1 = fig.add_subplot(211)
ax1.plot(range(1, len(rho)+1), rho, 'o-')
ax1.set_title('Variance explained by principal components')
ax1.set_xlabel('Principal component')
ax1.set_ylabel('Variance explained')

ax2 = fig.add_subplot(212)
ax2.plot(range(1, len(cum_rho)+1), cum_rho, 'o-')
ax2.axhline(y=0.9, color='r', linestyle='--')
ax2.set_title('Cumulative variance explained by principal components')
ax2.set_xlabel('Principal component')
ax2.set_ylabel('Cumulative variance explained')
show()


list1, list2 = zip(*sorted(zip(attributeNames, V[:, 1]), key=lambda x: x[1]))
attr_sorted_by_value = dict(zip(list1,  list2))
print(attr_sorted_by_value)

list1, list2 = zip(*sorted(zip(attributeNames, V[:, 2]), key=lambda x: x[1]))
attr_sorted_by_value = dict(zip(list1,  list2))
print(attr_sorted_by_value)

# Project data onto principal component space
Y = X - X.mean(axis=0)
Z = X @ V

n = [(0, 250000), (250000, 350000), (350000, 500000), (500000, 700000)]

# Plot PCA of the data
f = figure()
title('pixel vectors of handwr. digits projected on PCs')
for c in n:
    # select indices belonging to class c:
    class_mask1 = (y >= c[0])
    class_mask2 = (y < c[1])
    class_mask = (class_mask1 & class_mask2)
    print(type(class_mask))
    print(type(Z))
    plot(Z[class_mask, 0], Z[class_mask, 1], '.')
xlabel('PC1')
ylabel('PC2')
show()


figure()
plot(X)
