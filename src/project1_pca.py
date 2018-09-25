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

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()
cum_rho = rho.cumsum()

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

print('Ran Exercise 2.1.3')

list1, list2 = zip(*sorted(zip(attributeNames, V[:, 1]), key=lambda x: x[1]))
attr_sorted_by_value = dict(zip(list1,  list2))
print(attr_sorted_by_value)

list1, list2 = zip(*sorted(zip(attributeNames, V[:, 1]), key=lambda x: abs(x[1])))
attributes_sorted_by_abs_value = dict(zip(list1,  list2))
print(attributes_sorted_by_abs_value)