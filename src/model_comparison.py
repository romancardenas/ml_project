#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:44:32 2018
comparison of models using data saved before

@author: matthewparker
"""
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection, tree
from scipy import stats
import pandas as pd
import matplotlib as plt

plt.rcParams.update({'font.size': 16})


# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 

K = 10
data = pd.read_csv('../data/errors.csv')
Error_dectree = np.empty((K,1))
Error_knn = np.empty((K,1))
Error_baseline = np.empty((K,1))


Error_d = np.array(data['error1'])
Error_k = np.array(data['error2'])
Error_b = np.array(data['error3'])
Z_stats = np.empty((2,3))

for x in range(0,10):
    Error_dectree[x] = Error_d[x]
    Error_knn[x] = Error_k[x]
    Error_baseline[x] = Error_b[x]



z = (Error_dectree-Error_baseline)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);
Z_stats[0,0] = zL
Z_stats[1,0] = zH

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure(1)
boxplot(np.concatenate((Error_dectree, Error_baseline),axis=1))
xlabel('Decision Tree   vs.   Baseline')
ylabel('Cross-validation error [%]')



z = (Error_knn-Error_baseline)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

Z_stats[0,1] = zL
Z_stats[1,1] = zH

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure(2)
boxplot(np.concatenate((Error_knn, Error_baseline),axis=1))
xlabel('KNN   vs.   Baseline')
ylabel('Cross-validation error [%]')



z = (Error_knn-Error_dectree)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

Z_stats[0,2] = zL
Z_stats[1,2] = zH

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure(3)
boxplot(np.concatenate((Error_knn, Error_dectree),axis=1))
xlabel('KNN  vs.   Decision Tree')
ylabel('Cross-validation error [%]')


show()