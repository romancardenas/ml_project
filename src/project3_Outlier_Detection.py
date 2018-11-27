#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 12:32:58 2018

@author: sebastian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from matplotlib.pyplot import figure, subplot, hist, title, show, plot,bar
from scipy.stats.kde import gaussian_kde
from sklearn import preprocessing
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

## Load data file and extract variables of interest
##data = pd.read_csv('../data/kc_house_data_project_3.csv')
##data = pd.read_csv('../data/SAheart.csv')
data = pd.read_csv('../data/SAheart_reg.csv')
#data = data.replace(to_replace='Present',value='1', inplace=False, limit=None, regex=False, method='pad')
#data = data.replace(to_replace='Absent' ,value='0', inplace=False, limit=None, regex=False, method='pad')
#
#data = data.drop(['tobacco','alcohol','sbp'], axis=1)
A = data.values[:, :]
#print(type(X))

X = preprocessing.scale(A)
#X = scaler.transform(X)
#standardize


attributeNames = list(data)[:]
classNames = list(data)[:]
N, M = X.shape
C = len(classNames)


## Example Dataset of them
## Draw samples from mixture of gaussians (as in exercise 11.1.1)
#N = 1000; M = 8
#x = np.linspace(-10, 10, 50)
#X = np.empty((N,M))
#m = np.array([1, 3, 6]); s = np.array([1, .5, 2])
#c_sizes = np.random.multinomial(N, [1./3, 1./3, 1./3])
#for c_id, c_size in enumerate(c_sizes):
#    X[c_sizes.cumsum()[c_id]-c_sizes[c_id]:c_sizes.cumsum()[c_id],:] = np.random.normal(m[c_id], np.sqrt(s[c_id]), (c_size,M))



#y_one_zipcode = np.empty((y.shape[0], 1))
#for column in range(C):
#    aux = (column + 1) * y[:, [column]]
#    y_one_zipcode += (column + 1) * y[:, [column]]

###############################################################################
############ Kernel density outlier score
#############################################################################
## Compute kernel density estimate
#kde = gaussian_kde(X.ravel())
#
#scores = kde.evaluate(X.ravel())
#idx = scores.argsort()
#scores.sort()
#
#print('The index of the lowest density object: {0}'.format(idx[0]))
#
## Plot kernel density estimate
#figure(0)
#bar(range(50),scores[:50])
#title('Outlier score')
#show()
#############################################################################
############ end Kernel density outlier score
##############################################################################

#############################################################################
############ Kernel Density
##############################################################################

# Estimate the optimal kernel density width, by leave-one-out cross-validation
widths = 2.0**np.arange(-10,10)
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
    f, log_f = gausKernelDensity(X, w)
    logP[i] = log_f.sum()
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# Estimate density for each observation not including the observation
# itself in the density estimate
density, log_density = gausKernelDensity(X, width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i]




#############################################################################
############ END Kernel Density
##############################################################################

#############################################################################
############ KNN Density
##############################################################################

K = 10

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Compute the density
#D, i = knclassifier.kneighbors(np.matrix(xe).T)
knn_density = 1./(D.sum(axis=1)/K)


# Compute the average relative density
DX, iX = knn.kneighbors(X)
knn_densityX = 1./(DX[:,1:].sum(axis=1)/K)
knn_avg_rel_density = knn_density/(knn_densityX[i[:,1:]].sum(axis=1)/K)

i = (knn_density.argsort(axis=0)).ravel()
knn_density=knn_density[i]

j = (knn_avg_rel_density.argsort(axis=0)).ravel()
knn_avg_rel_density = knn_avg_rel_density[j]


#############################################################################
############ END KNN Density
##############################################################################

#############################################################################
############ Print results
##############################################################################


for it in range(10):  
    # Display the index of the lowest density data object
    print('Lowest density: {0} for data object: {1}'.format(density[it],i[it]))
    # Display the index of the lowest density data object
    print('Lowest knn_density: {0} for data object: {1}'.format(knn_density[it],i[it]))
    # Display the index of the lowest density data object
    print('Lowest knn_avg_rel_density: {0} for data object: {1}'.format(knn_avg_rel_density[it],i[it]))

# Plot density estimate of outlier score
figure(1)
bar(range(50),density[:50].reshape(-1,))
title('Density estimate')
figure(2)
plot(logP)
title('Optimal width')
# Plot density estimate of outlier score
figure(3)
bar(range(50),knn_density[:50].reshape(-1,))
title('knn_Density_estimate')
figure(4)
bar(range(50),knn_avg_rel_density[:50].reshape(-1,))
title('knn_avg_rel_density_estimate')
show()


#############################################################################
############ Print results
##############################################################################


