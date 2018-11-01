#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:40:08 2018

@author: matthewparker
"""

import pandas as pd
import numpy as np
from sklearn import tree
import graphviz

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


# Fit regression tree classifier, Gini split criterion, no pruning
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=2)
dtc = dtc.fit(X,y)

# Export tree graph for visualization purposes:
# (note: you can use i.e. Graphviz application to visualize the file)
out = tree.export_graphviz(dtc, out_file='tree_gini.gvz', feature_names=attributeNames)
#graphviz.render('dot','png','tree_gini',quiet=False)
src=graphviz.Source.from_file('tree_gini.gvz')
## Comment in to automatically open pdf
## Note. If you get an error (e.g. exit status 1), try closing the pdf file/viewer
src.render('../tree_gini', view=True)

print('Done')