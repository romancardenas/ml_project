#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:40:08 2018

@author: matthewparker
"""

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
from scipy.stats import mode


data = pd.read_csv('../data/kc_house_data_clean_regularzip.csv')

dataObjectNames = np.array(range(0,7045))

attributeNames = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']

X = np.asarray(data[['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']])

allCNames = np.array(data['zipcode'])

classNames = np.unique((data['zipcode']))

cnTemp = classNames.tolist()

y = list(range(0,7045))
for x in range(7045):
    y[x]=cnTemp.index(allCNames[x])

N, M = X.shape
C = len(classNames)

mode_zip = mode(allCNames)
print('Predicting the most occuring zip yields an error of: {0}'.format(1-mode_zip[1]/7045))

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)

# Fit regression tree classifier, Gini split criterion, no pruning
dtc = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2, max_depth = 5)
dtc = dtc.fit(X_train,y_train)
predicted = dtc.predict(X_test)
i = 0;
n = len(predicted)
for x in range(n):
    if predicted[x] == y_test[x]:
        i=i+1

print('The error of this model is: {0} '.format(1-i/n))






