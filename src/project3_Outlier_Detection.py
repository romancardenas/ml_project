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
from sklearn.preprocessing import Normalizer
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

# Load data file and extract variables of interest
#data = pd.read_csv('../data/kc_house_data_project_3.csv')
data = pd.read_csv('/Users/sebastian/Desktop/test.csv')
X = data.values[:5000, :6]

scaler = Normalizer().fit(X)
X = scaler.transform(X)

y = data.values[:5000, 6:]


attributeNames = list(data)[:6]
classNames = list(data)[6:]
N, M = X.shape
C = len(classNames)
y_one_zipcode = np.empty((y.shape[0], 1))
for column in range(C):
    aux = (column + 1) * y[:, [column]]
    y_one_zipcode += (column + 1) * y[:, [column]]

##############################################################################
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

# Display the index of the lowest density data object
print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))

# Plot density estimate of outlier score
figure(1)
bar(range(50),density[:50].reshape(-1,))
title('Density estimate')
figure(2)
plot(logP)
title('Optimal width')
show()


#############################################################################
############ END Kernel Density
##############################################################################

#############################################################################
############ KNN Density
##############################################################################

x = np.linspace(0, 69, 69)

# Number of neighbors
K = 100

# x-values to evaluate the KNN
xe = np.linspace(0, 69, 69)

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Compute the density
#D, i = knclassifier.kneighbors(np.matrix(xe).T)
knn_density = 1./(D.sum(axis=1)/K)

# Compute the average relative density
DX, iX = knn.kneighbors(X)
knn_densityX = 1./(DX[:,1:].sum(axis=1)/K)
knn_avg_rel_density = knn_density/(knn_densityX[i[:,1:]].sum(axis=1)/K)


# Plot KNN density
figure(figsize=(6,7))
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(y, knn_density)
title('KNN density')
# Plot KNN average relative density
figure(figsize=(6,7))
subplot(2,1,1)
hist(X,x)
title('Data histogram')
subplot(2,1,2)
plot(y, knn_avg_rel_density)
title('KNN average relative density')

show()
#############################################################################
############ END KNN Density
##############################################################################

