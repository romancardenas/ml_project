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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import MultinomialNB



data = pd.read_csv('../data/kc_house_data_clean_regression_nozip.csv')

dataObjectNames = np.array(range(0,7045))

attributeNames = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']

X = np.asarray(data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']])

scaler = Normalizer().fit(X)
X = scaler.transform(X)

#X = StandardScaler().fit_transform(X)

N, M = X.shape



allCNames = np.array(data['price'])
allCNames.sort()
plot(allCNames)


low = 4574
med = 9167

priceCategories = list(range(0,N))
i = 0

counter_low=0
counter_med_low=0
counter_med_high=0
counter_high=0


#for x in range(0,N):
#    if x <N/4:
#        priceCategories[x] = 'low'
#        counter_low+=1
#    elif x <N/2:
#        priceCategories[x] = 'med_low'
#        counter_med_low+=1
#    elif x <(3*N)/4:
#        priceCategories[x] = 'med_high'
#        counter_med_high+=1
#    else:
#        priceCategories[x] = 'high'
#        counter_high+=1
for x in allCNames:
    if x < 280000:
        priceCategories[i] = 'low'
        counter_low+=1
    elif x < 400000:
        priceCategories[i] = 'med_low'
        counter_med_low+=1
    elif x < 600000:
        priceCategories[i] = 'med_high'
        counter_med_high+=1
    else :
        priceCategories[i] = 'high'
        counter_high+=1
    i+=1
#    
#    
#    
####################
# count the number of houses in each category
#####
#counter_low=0
#counter_med=0
#counter_high=0
#
#for i in range(N):
#      if df.at[i, 'price']<border_low:
#          counter_low=counter_low+1
#      elif df.at[i, 'price']>border_low and df.at[i, 'price']<border_med:
#          counter_med=counter_med+1
#      else:
#          counter_high=counter_high+1
#      i+=1

####
####################
      
      
classNames = ['low','med_low','med_high','high']
C = len(classNames)

y = np.asarray(list(range(0,N)))
for x in range(N):
    y[x] = classNames.index(priceCategories[x])






###NOTE: In order to save time I just created 3 tc variables, uncomment the appropriate one 
### depending on which parameter you wish to control

#depth as complexity controlling
tc = np.arange(2, 21, 1)
name = 'depth'

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
dtc_Error_train = np.empty((K,1))
dtc_Error_test = np.empty((K,1))
dtc_optimal_depth = np.empty((K,1))

k=0
col = 0;
row = 0;
L = 50
alpha = 1.0


# Initialize variable
knn_Error_test = np.empty((K,1))
knn_optimal_N = np.empty((K,1))


#NB Errors
NB_Error_test = np.empty((K,1))
#AllErrors = [[range(0,K+1)][range(0,7)]
#AllErrors[0,:] = ['DTC TEST ERROR', 'DTC TRAIN ERROR', 'Depth', 'KNN ERROR', 'Neighbors', 'NB ERROR', 'Prior']


#f, ax = plt.subplots(4,3)
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    

    K2 = 10
    CV2 = model_selection.KFold(n_splits=K,shuffle=True)
    dtc_inner_error_train = np.empty((len(tc), K2))
    dtc_inner_error_test = np.zeros((len(tc), K2))
    k2=0
    knn_errors = np.empty((K2,L))
    
    nb_errors = np.empty((K2,2))
    

    for dtrain_index, dval_index in CV2.split(X_train):
        print('Computing inner CV fold: {0}/{1}..'.format(k2+1,K2))
        
        dtrainx, dtrainy = X_train[dtrain_index,:], y[dtrain_index]
        dtestx, dtesty = X_train[dval_index,:], y[dval_index]
        
        
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(dtrainx, dtrainy);
            y_est = knclassifier.predict(dtestx);
#            accuracy=accuracy_score(dtesty,y_est)
#            print('accuracy is ',accuracy)
            knn_errors[k2,l-1] = np.sum(y_est!=dtesty)/len(y_est)
            

        
        for i, t in enumerate(tc):
             dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth = t)
             dtc = dtc.fit(dtrainx,dtrainy.ravel())
             y_est_test = dtc.predict(dtestx)
             y_est_train = dtc.predict(dtrainx)
             misclass_rate_test = sum((y_est_test!=dtesty)) / float(len(y_est_test))
             misclass_rate_train = sum((y_est_train!= dtrainy)) /float(len(y_est_train))
             dtc_inner_error_test[i,k2], dtc_inner_error_train[i,k2] = misclass_rate_test, misclass_rate_train
             
        nb_classifier_uni = MultinomialNB(alpha=alpha, fit_prior=False)
        nb_classifier_uni.fit(dtrainx, dtrainy)
        y_est_prob_uni = nb_classifier_uni.predict_proba(dtestx)
        y_est_uni = np.argmax(y_est_prob_uni,1)
        
        nb_classifier_est = MultinomialNB(alpha=alpha, fit_prior=True)
        nb_classifier_est.fit(dtrainx, dtrainy)
        y_est_prob_est = nb_classifier_est.predict_proba(dtestx)
        y_est_est = np.argmax(y_est_prob_est,1)
    

        nb_errors[k2,0] = np.sum(y_est_uni!=dtesty,dtype=float)/dtesty.shape[0]
        nb_errors[k2,1] = np.sum(y_est_est!=dtesty,dtype=float)/dtesty.shape[0]
        
        
        k2+=1
        
        

    
    knn_index = 0;
    knn_means = np.mean(knn_errors, axis = 0)
    knn_best = knn_means[0]
    for j in range(1,L):
        if knn_means[j] < knn_best:
            knn_best = knn_means[j]
            knn_index = j    
    
    # get the index and lowest error
    dtc_index = 0;
    dtc_means = np.mean(dtc_inner_error_test, axis = 1)
    dtc_best = dtc_means[0]
    for j in range(1,len(tc)):
        if dtc_means[j] < dtc_best:
            dtc_best = dtc_means[j]
            dtc_index = j

     
    nb_means = np.mean(nb_errors, axis = 0)
    if nb_means[0] < nb_means[1]:
        fin_prior = True
        p = "uniform"
        z = 0
    else:
        fin_prior = False
        p = "estimate"
        z = 1
    
    print('dtc_test_Error of {0} with {1} of {2}'.format(misclass_rate_test,name, tc[dtc_index]))
    

            
    print("knn_index : {0} selected as best with error of {1}".format(knn_index, knn_best))
#    figure()
#    plot(100*(knn_means))
#    xlabel('Number of neighbors')
#    ylabel('Classification error rate (%)')
#    show()
        
    print("mb prior selected: {0} with error rate of {1}".format(p, nb_means[z]))
    
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth = tc[dtc_index])
    dtc = dtc.fit(X_train,y_train.ravel())
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    # Evaluate misclassification rate over train/test data (in this CV fold)
    misclass_rate_test = sum((y_est_test!=y_test)) / float(len(y_est_test))
    misclass_rate_train = sum((y_est_train!=y_train)) / float(len(y_est_train))
    dtc_Error_test[k], dtc_Error_train[k] = misclass_rate_test, misclass_rate_train
    
    
    knclassifier = KNeighborsClassifier(n_neighbors=knn_index+1);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    knn_Error_test[k] = np.sum(y_est!=y_test)/len(y_est)
    
    
    nb_classifier_final = MultinomialNB(alpha=alpha, fit_prior=fin_prior)
    nb_classifier_final.fit(X_train, y_train)
    y_est_prob_final = nb_classifier_final.predict_proba(X_test)
    y_est_final = np.argmax(y_est_prob_final,1)
    
    
    NB_Error_test[k] = np.sum(y_est_final!=y_test,dtype=float)/y_test.shape[0]
    
#    AllErrors[k,:] = [misclass_rate_test, misclass_rate_train, dtc_index, knn_Error_test[k], knn_index+1, NB_Error_test[k], fin_prior]
    k+=1


#index = 0;
#means = np.mean(Error_test, axis = 1)
#best = means[0]
#for j in range(1,K):
#    if means[j] < best:
#        best = means[j]
#        index = j
        

#print('Best error of: {0} at a depth of : {1}'.format(best, optimal_depth[index]))












