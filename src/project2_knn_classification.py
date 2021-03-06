# exercise 7.1.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('../data/kc_house_data_clean_regression_zip.csv')

dataObjectNames = np.array(range(0,7045))

attributeNames = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']

X = np.asarray(data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement']])

X = StandardScaler().fit_transform(X)

allCNames = np.array(data['price'])
allCNames.sort()

z = np.mean(allCNames)

low = 4574
med = 9167

priceCategories = list(range(0,13724))
i = 0
low = 0
med = 0
high = 0
for x in allCNames:
    if x < 400000:
        priceCategories[i] = 'low'
        low += 1
    elif x < 600000:
        priceCategories[i] = 'med'
        med += 1
    else :
        priceCategories[i] = 'high'
        high +=1
    i+=1
    
classNames = ['low','med','high']

#y = np.asarray(list(range(0,13724)))
#for x in range(13724):
#    y[x] = classNames.index(priceCategories[x])
#
#
#N, M = X.shape
#C = len(classNames)
#
#
## Maximum number of neighbors
## K-fold crossvalidation
#K = 10
#L = 40
#CV = model_selection.KFold(n_splits=K,shuffle=True)
#
## Initialize variable
#Error_test = np.empty((K,1))
#optimal_depth = np.empty((K,1))
#
#k=0
#col = 0;
#row = 0;
#
#for train_index, test_index in CV.split(X):
#    print('Computing CV fold: {0}/{1}..'.format(k+1,K))
#
#    # extract training and test set for current CV fold
#    X_train, y_train = X[train_index,:], y[train_index]
#    X_test, y_test = X[test_index,:], y[test_index]
#    
#
#    K2 = 10
#    CV2 = model_selection.KFold(n_splits=K,shuffle=True)
#    k2 = 0
#    errors = np.zeros((K2,L))
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
#            errors[k2,l-1] = np.sum(y_est!=dtesty)/len(y_est)
#        k2+=1
#        
#    index = 0;
#    sums = np.mean(errors, axis = 0)
#    best = sums[0]
#    for j in range(1,L):
#        if sums[j] < best:
#            best = sums[j]
#            index = j
#    print("index : {0} selected as best with error of {1}".format(index, best))
#    figure()
#    plot(100*(sums))
#    xlabel('Number of neighbors')
#    ylabel('Classification error rate (%)')
#    show()
#    
#    knclassifier = KNeighborsClassifier(n_neighbors=index+1);
#    knclassifier.fit(X_train, y_train);
#    y_est = knclassifier.predict(X_test);
#    Error_test[k] = np.sum(y_est!=y_test)/len(y_est)
#    k+=1
#        
#    
#
#
#    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
#
#
#
#print('Ran Exercise 7.1.2')