#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:18:39 2018

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
K = 15
L = 40
dist = 1

#CV = model_selection.KFold(n_splits=K,shuffle=False)

#### test Basti
#####
#####
#####
errors = np.zeros((N,L))
i=0

#CV.split(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle=False)

knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)

#cm = confusion_matrix(y_test, y_est);
#accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
accuracy = accuracy_score(y_test, y_est)
error_rate = 1-accuracy
#error-rate = 1-accuracy
print(accuracy)
print(error_rate)
######
######
######

#for train_index, test_index in CV.split(X, y):
#    print('Crossvalidation fold: {0}/{1}'.format(i+1,K))  
#    
#    X_train = X[train_index,:]
#    y_train = y[train_index]
#    X_test = X[test_index,:]
#    y_test = y[test_index]
#    
#    for l in range (1, L+1):
#        knclassifier = KNeighborsClassifier(n_neighbors=l)
#        knclassifier.fit(X_train, y_train);
#        y_est = knclassifier.predict(X_test);
#        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
#    i+=1
#    
#figure()
#plot(sum(errors,0)/N)
#xlabel('Number of neighbors')
#ylabel('Classification error rate (%)')
#show()

#### end test Basti

# Initialize variable
#Error_test = np.empty((K,1))
#optimal_depth = np.empty((K,1))

#k=0
#col = 0;
#row = 0;

#f, ax = plt.subplots(4,3)
#for train_index, test_index in CV.split(X):
#    print('Computing CV fold: {0}/{1}..'.format(k+1,K))
#
#    # extract training and test set for current CV fold
#    X_train, y_train = X[train_index,:], y[train_index]
#    X_test, y_test = X[test_index,:], y[test_index]
#    figure()
#    plot(X_train)
#    
#    K2 = 10
#    CV2 = model_selection.KFold(n_splits=K,shuffle=True)
#    k2 = 0
#    errors = np.empty((K2,L))
#    
#    for dtrain_index, dval_index in CV2.split(X_train):
#        print('Computing inner CV fold: {0}/{1}..'.format(k2+1,K2))
#        
#        dtrainx, dtrainy = X_train[dtrain_index,:], y[dtrain_index]
#        dtestx, dtesty = X_train[dval_index,:], y[dval_index]
#    
#        for l in range(1,L+1):
#            knclassifier = KNeighborsClassifier(n_neighbors=l);
#            knclassifier.fit(dtrainx, dtrainy);
#            y_est = knclassifier.predict(dtestx);
#            errors[k2,l-1] = np.sum(y_est[0]!=dtesty[0])
#        k2+=1
#        
#    index = 0;
#    sums = np.sum(errors, axis = 0)
#    best = sums[0]
#    for j in range(1,L):
#        if sums[j] < best:
#            best = sums[j]
#            index = j
#    figure()
#    plot(100*sum(errors,0)/len(y_est))
#    xlabel('Number of neighbors')
#    ylabel('Classification error rate (%)')
#    show()
#    
#    knclassifier = KNeighborsClassifier(n_neighbors=index+1);
#    knclassifier.fit(X_train, y_train);
#    y_est = knclassifier.predict(X_test);
#    Error_test[k] = np.sum(y_est!=y_test)
#    k+=1
        
    


    # Fit classifier and classify the test points (consider 1 to 40 neighbors)



#print('Ran Exercise 7.1.2')