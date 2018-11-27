import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
###################################################
#                   READ DATA                     #
###################################################
data = pd.read_csv('../data/kc_house_data_project_3_reg.csv')
data = data.drop(['price', 'sqft_living', 'sqft_lot', 'yr_built'], axis=1)
X = data.values[:, :2]
X[:, [0, 1]] = X[:, [1, 0]]
y = data.values[:, 2:]
attributeNames = list(data)[:2]
classNames = list(data)[2:]
N, M = X.shape
C = len(classNames)
y_one_zipcode = np.empty((y.shape[0], 1))
for column in range(C):
    aux = (column + 1) * y[:, [column]]
    y_one_zipcode += (column + 1) * y[:, [column]]

print('###################################################')
print('#                        GMM                      #')
print('###################################################')

# Range of K's to try
KRange = range(C - 10, C + 10)  # TODO
T = len(KRange)

covar_type = 'full'  # you can try out 'diag' as well
reps = 3  # number of fits with different initalizations, best result will be kept

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10, shuffle=True)

for t, K in enumerate(KRange):
    print('Fitting model for K={0}'.format(K))

    # Fit Gaussian mixture model
    gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X)

    # Get BIC and AIC
    BIC[t] = gmm.bic(X)
    AIC[t] = gmm.aic(X)

    # For each crossvalidation fold
    for train_index, test_index in CV.split(X):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        X_test = X[test_index]

        # Fit Gaussian mixture model to X_train
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

        # compute negative log likelihood of X_test
        CVE[t] += -gmm.score_samples(X_test).sum()

# Plot results
plt.figure(1, figsize=(20, 10))
plt.plot(KRange, BIC, '-*b')
plt.plot(KRange, AIC, '-xr')
plt.plot(KRange, 2 * CVE, '-ok')
plt.legend(['BIC', 'AIC', 'Crossvalidation'])
plt.xlabel('K')
plt.title('GMM Cross-Validation Error')
plt.show()

# Select best value for K provided the GMM
index_of_max = np.asscalar(np.argmin(CVE))
K_optimal = KRange[index_of_max]
print('The optimal number of clusters, according to GMM cross-validation, is {}'.format(K_optimal))

# Fit best Gaussian mixture model to X and plot result
gmm = GaussianMixture(n_components=K_optimal, covariance_type=covar_type, n_init=reps).fit(X)
cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
plt.figure(2, figsize=(24, 18))
plt.title('Gaussian Mixture Model using {} clusters'.format(K_optimal))
plt.xlabel('longitude')
plt.ylabel('latitude')
# aux = X[:, [attributeNames.index('long'), attributeNames.index('lat')]]
clusterplot(X, clusterid=cls, centroids=cds, y=y_one_zipcode, covars=covs)
plt.show()

print('###################################################')
print('#             HIERARCHICAL CLUSTERING             #')
print('###################################################')
Method = 'average'
Methods = ['single', 'complete', 'average', 'weighted', 'median', 'ward']
Metric = 'euclidean'
# for Method in Methods: TODO
Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = K_optimal
cls = fcluster(Z, criterion='maxclust', t=Maxclust)

plt.figure(3, figsize=(24, 18))
plt.title('Hierarchical clustering using {} method'.format(Method))
plt.xlabel('longitude')
plt.ylabel('latitude')
# aux = X[:, [attributeNames.index('long'), attributeNames.index('lat')]]
# aux = (aux - aux.mean(axis=0))/aux.std(axis=0)
clusterplot(X, cls, y=y_one_zipcode)

# Display dendrogram
max_display_levels = K_optimal
plt.figure(4, figsize=(20, 10))
plt.title('Hierarchical clustering using {} method'.format(Method))
dendrogram(Z, truncate_mode='lastp', p=max_display_levels)
plt.show()

print('###################################################')
print('#            MODELS QUALITY EVALUATION            #')
print('###################################################')
# TODO
