#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:06:25 2018

@author: sebastian
"""


from matplotlib.pyplot import (figure, hold, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('../data/kc_house_data_clean_regularzip.csv')

dataObjectNames = np.array(range(0,7045))

attributeNames = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']

X = np.asarray(data[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']])
#### standardize the data ####
X = StandardScaler().fit_transform(X)
#### data standardized   ####
allCNames = np.array(data['zipcode'])

classNames = np.unique((data['zipcode']))

cnTemp = classNames.tolist()

y = np.asarray(list(range(0,7045)))
for x in range(7045):
    y[x]=cnTemp.index(allCNames[x])

N, M = X.shape
C = len(classNames)


# Maximum number of neighbors
# K-fold crossvalidation
K = 10
L = 40
i=0
dist = 1

CV = model_selection.KFold(n_splits=K)
#CV = model_selection.LeaveOneOut()
errors = np.zeros((N,L))
i=0
for train_index, test_index in CV.split(X):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,K))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    i+=1
    
# Plot the classification error rate
#figure()
#plot(100*sum(errors,0)/N)
#xlabel('Number of neighbors')
#ylabel('Classification error rate (%)')
#show()

#error_sum = 0
#for i in range(0,len(y_test)-1):
#    if y_test[i]!=y_est[i]:
#        error_sum = error_sum+1
#print(error_sum)
#accuracy_of_CV = 1 - error_sum/len(y_est)
#print(accuracy_of_CV)