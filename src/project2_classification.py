#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:40:08 2018

@author: matthewparker
"""

import pandas as pd
import numpy as np
import graphviz
from scipy.stats import mode
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from scipy.io import loadmat
from sklearn import model_selection, tree
import numpy as np



data = pd.read_csv('../data/kc_house_data_clean_regularzip.csv')

dataObjectNames = np.array(range(0,7045))

attributeNames = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']

X = np.asarray(data[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']])

allCNames = np.array(data['zipcode'])

classNames = np.unique((data['zipcode']))

cnTemp = classNames.tolist()

y = np.asarray(list(range(0,7045)))
for x in range(7045):
    y[x]=cnTemp.index(allCNames[x])

N, M = X.shape
C = len(classNames)

tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
optimal_depth = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    

    K2 = 10
    CV2 = model_selection.KFold(n_splits=K,shuffle=True)
    inner_error_train = np.empty((len(tc), K2))
    inner_error_test = np.empty((len(tc), K2))
    k2=0
    for dtrain_index, dval_index in CV.split(X_train):
        print('Computing inner CV fold: {0}/{1}..'.format(k2+1,K2))
        
        dtrainx, dtrainy = X_train[dtrain_index,:], y[dtrain_index]
        dtestx, dtesty = X_train[dval_index,:], y[dval_index]
        
        for i, t in enumerate(tc):
             dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
             dtc = dtc.fit(dtrainx,dtrainy.ravel())
             y_est_test = dtc.predict(dtestx)
             y_est_train = dtc.predict(dtrainx)
             misclass_rate_test = sum(np.abs(y_est_test - dtesty)) / float(len(y_est_test))
             misclass_rate_train = sum(np.abs(y_est_train - dtrainy)) /float(len(y_est_train))
             inner_error_test[i,k2], inner_error_train[i,k2] = misclass_rate_test, misclass_rate_train
        k2+=1
        
    f = figure()
    plot(tc, inner_error_train.mean(1))
    plot(tc, inner_error_test.mean(1))
    xlabel('Model complexity (max tree depth)')
    ylabel('Error (misclassification rate, CV K={0})'.format(K))
    legend(['Error_train','Error_test'])
    
    show()
        
    # get the index and lowest error
    index = 0;
    means = np.mean(inner_error_test, axis = 1)
    best = means[0]
    for j in range(1,len(tc)):
        if means[j] < best:
            best = means[j]
            index = j
    
        
    
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth = tc[index])
    dtc = dtc.fit(X_train,y_train.ravel())
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))
    misclass_rate_train = sum(np.abs(y_est_train - y_train)) / float(len(y_est_train))
    Error_test[k], Error_train[k] = misclass_rate_test, misclass_rate_train
    optimal_depth[k] = tc[index]
    k+=1
    print('Error of {0} with depth of {1}'.format(misclass_rate_test, tc[index]))
    
    
index = 0;
means = np.mean(Error_test, axis = 1)
best = means[0]
for j in range(1,K):
    if means[j] < best:
        best = means[j]
        index = j
        

print('Best error of: {0} at a depth of : {1}'.format(best, optimal_depth[index]))









