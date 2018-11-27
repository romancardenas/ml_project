import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from toolbox_02450 import clusterval
from scipy.linalg import svd

###################################################
#                   READ DATA                     #
###################################################
data = pd.read_csv('../data/seed_data_reg.csv')
X = data.values[:, 1:]
y = data.values[:, [0]].ravel()
attributeNames = list(data)[1:]
classNames = list(data)[0]
N, M = X.shape
C = len(classNames)

Y = (X - X.mean(axis=0))
# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)
V = V.T
X_projected = Y @ V
X = X_projected[:, :2]

print('###################################################')
print('#                        GMM                      #')
print('###################################################')

# Range of K's to try
KRange = range(1, 10)  # TODO
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
plt.figure(2, figsize=(12, 9))
plt.title('Gaussian Mixture Model using {} clusters'.format(K_optimal))
plt.xlabel('PC 1')
plt.ylabel('PC 2')
clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
plt.show()

# Evaluate GMM model
Rand_gmm, Jaccard_gmm, NMI_gmm = clusterval(y, cls)

print('###################################################')
print('#             HIERARCHICAL CLUSTERING             #')
print('###################################################')
Metric = 'euclidean'
Maxclust = K_optimal
max_display_levels = K_optimal
Methods = ['single', 'complete', 'average', 'weighted', 'median', 'ward']  # We will try all these methods
n_methods = len(Methods)

# Allocate variables:
Rand_hier = np.zeros((n_methods,))
Jaccard_hier = np.zeros((n_methods,))
NMI_hier = np.zeros((n_methods,))

i = 0
for Method in Methods:
    Z = linkage(X, method=Method, metric=Metric)

    # Compute and display clusters by thresholding the dendrogram
    cls = fcluster(Z, criterion='maxclust', t=Maxclust)

    plt.figure(3 + 2*i, figsize=(12, 9))
    plt.title('Hierarchical clustering using {} method'.format(Method))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    clusterplot(X, cls, y=y)

    # Display dendrogram
    plt.figure(4 + 2*i, figsize=(15, 8))
    plt.title('Hierarchical clustering using {} method'.format(Method))
    dendrogram(Z, truncate_mode='lastp', p=max_display_levels)
    plt.show()

    # Evaluate hierarchical method
    Rand_hier[i], Jaccard_hier[i], NMI_hier[i] = clusterval(y, cls)
    i += 1


print('###################################################')
print('#            MODELS QUALITY EVALUATION            #')
print('###################################################')
Jaccard = {'gmm': Jaccard_gmm}
NMI = {'gmm': NMI_gmm}
Rand = {'gmm': Rand_gmm}
for i in range(len(Methods)):
    Jaccard[Methods[i]] = Jaccard_hier[i]
    NMI[Methods[i]] = NMI_hier[i]
    Rand[Methods[i]] = Rand_hier[i]
pass

print('Jaccard: {}'.format(str(Jaccard)))
print('NMI: {}'.format(str(NMI)))
print('Rand: {}'.format(str(Rand)))
print('###################################################')
print('GMM centroids: {}'.format(str(cds)))
print('GMM covariances: {}'.format(str(covs)))