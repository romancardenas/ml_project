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
from matplotlib.pyplot import figure, subplot, hist, title, show, plot, bar, xlabel, ylabel, xticks
from scipy.stats.kde import gaussian_kde
from sklearn import preprocessing
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

## Load data file and extract variables of interest
data = pd.read_csv('../data/seed_data_reg.csv')

#data = data.drop(['tobacco','alcohol','sbp'], axis=1)
X = data.values[:, :]
#print(type(X))

#standardize data
X = preprocessing.scale(X)

attributeNames = list(data)[:]
classNames = list(data)[:]
N, M = X.shape
C = len(classNames)

#############################################################################
############ Kernel Density estimation
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
############ END Kernel Density estimation
##############################################################################

#############################################################################
############ KNN Density 
##############################################################################

K = 10

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, knn_i = knn.kneighbors(X)

# Compute the density
#D, i = knclassifier.kneighbors(np.matrix(xe).T)
knn_density = 1./(D.sum(axis=1)/K)


# Compute the average relative density
DX, iX = knn.kneighbors(X)
knn_densityX = 1./(DX[:,1:].sum(axis=1)/K)
knn_avg_rel_density = knn_density/(knn_densityX[iX[:,1:]].sum(axis=1)/K)

knn_i = (knn_density.argsort(axis=0)).ravel()
knn_density=knn_density[knn_i]


iX = (knn_avg_rel_density.argsort(axis=0)).ravel()
knn_avg_rel_density = knn_avg_rel_density[iX]


#############################################################################
############ END KNN Density
##############################################################################

#############################################################################
############ Print results
##############################################################################


for it in range(10):  
    # Display the index of the lowest density data object
    print('density: {0} for data object: {1}'.format(density[it],i[it]))
    # Display the index of the lowest density data object
    print('knn_density: {0} for data object: {1}'.format(knn_density[it],knn_i[it]))
    # Display the index of the lowest density data object
    print('knn_avg_rel_density: {0} for data object: {1}'.format(knn_avg_rel_density[it],iX[it]))

# Plot density estimate of outlier score
figure(1)
plot(logP)
title('Optimal width')
xlabel('log10(sigma^2)')
ylabel('logP(X)')
# Plot density estimate of outlier score
figure(2)
bar(range(186),density[:186].reshape(-1,))
title('kernel density estimate')
xticks(np.arange(186), i, rotation='vertical')
xlabel('index')
ylabel('density')
figure(3)
bar(range(186),knn_density[:186].reshape(-1,))
title('knn_Density_estimate')
xticks(np.arange(186), knn_i, rotation='vertical')
xlabel('index')
ylabel('knn_density')
figure(4)
bar(range(186),knn_avg_rel_density[:186].reshape(-1,))
title('knn_avg_rel_density_estimate')
xticks(np.arange(186), iX, rotation='vertical')
xlabel('index')
ylabel('knn_avg_rel_density')
show()


#############################################################################
############ END Print results
##############################################################################


