import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# Load data file and extract variables of interest
data = pd.read_csv('../data/kc_house_data_project_3.csv')
X = data.values[:, :6]
y = data.values[:, 6:]
attributeNames = list(data)[:6]
classNames = list(data)[6:]
N, M = X.shape
C = len(classNames)
y_one_zipcode = np.empty((y.shape[0], 1))
for column in range(C):
    aux = (column + 1) * y[:, [column]]
    y_one_zipcode += (column + 1) * y[:, [column]]


# Perform hierarchical/agglomerative clustering on data matrix
Method = 'average'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = C
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
plt.figure(1, figsize=(24, 18))
plt.title('Hierarchical clustering using {} method'.format(Method))
plt.xlabel('longitude')
plt.ylabel('latitude')
aux = X[:, [attributeNames.index('long'), attributeNames.index('lat')]]
# aux = (aux - aux.mean(axis=0))/aux.std(axis=0)
clusterplot(aux, cls, y=y_one_zipcode)

plt.figure(2, figsize=(24, 18))
plt.title('Hierarchical clustering using {} method'.format(Method))
plt.xlabel('longitude')
plt.ylabel('latitude')
for zipcode in range(1, C + 1):
    mask = y_one_zipcode[:, ] == zipcode
    mask_cool = np.zeros((N, M))
    hi = np.reshape(mask, (mask.shape[0], 1))
    mask_cool[:, attributeNames.index('long')] = np.reshape(mask, (mask.shape[0],))
    mask_cool[:, attributeNames.index('lat')] = np.reshape(mask, (mask.shape[0],))
    aux = X[mask_cool.astype(bool)]
    plt.plot(aux[:, [0]], aux[:, [1]], '.')
plt.show()

# Display dendrogram
max_display_levels = C
plt.figure(3, figsize=(20, 10))
plt.title('Hierarchical clustering using {} method'.format(Method))
dendrogram(Z, truncate_mode='lastp', p=max_display_levels)

plt.show()
