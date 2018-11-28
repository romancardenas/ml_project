""" Project 3 Associative mining
    Author: Matthew Parker
    Created on November 27th, 2018
    """
    
import numpy as np
import pandas as pd
from apyori import apriori


from similarity import binarize2

#data = pd.read_csv('../data/SAheart.csv', index_col = 0)
#
#data['famhist'] = (data['famhist'] == 'Present') * 1
#droplist = ['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age']
#data = divide_data(data, droplist)


def divide_data(df, labels, num_divides):
    """ divides data into quarters for better use with binarization """
    for x in labels:
        for i in range(num_divides):
            temp = df[x]
            #Qunatile ranges
            quant_low = i/num_divides
            quant_high = (i+1.0)/num_divides
            #Quantiles
            ql = temp.quantile(quant_low)
            qh = temp.quantile(quant_high)
            
            label = '_'+str(int(quant_low*100))+'-'+str(int(quant_high*100))
            
            if quant_low == (num_divides-1)/num_divides:
                df[x+label] = df[x]>=ql
            else:
                df[x+label] = ((df[x] >= ql) & (df[x] < qh))
            
            
#         quant = i/num_divides
#         q = temp.quantile(i/num_divides)
##        qml = temp.quantile(0.5)
##        qmh = temp.quantile(0.75)
#         newLabel = round(i/num_divides, 2)
            
#        df[x+'_0-25'] = df[x] < ql
#        df[x+'_25-50'] = ((df[x] >= ql) & (df[x] < qml))
#        df[x+'_50-75'] = ((df[x] >= qml) & (df[x] < qmh))
#        df[x+'_75-100'] = df[x] >= qmh
        
        df = df.drop(x,axis=1)
    
    return df * 1


data = pd.read_csv('../data/Seed_Data.csv',index_col = 0)
data = divide_data(data, list(data),10)




attributeNames = list(data)



N = len(data)
Y = len(attributeNames)

X = data.values[:,:]


#Xbin, attributeNamesBin = binarize2(X, attributeNames)


def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T

def print_apriori_rules(rules):
    frules = []
    for r in rules:
        conf = r.ordered_statistics[0].confidence
        supp = r.support
        x = ", ".join( list( r.ordered_statistics[0].items_base ) )
        y = ", ".join( list( r.ordered_statistics[0].items_add ) )
        print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
        frules.append( (x,y) )
    return frules




T = mat2transactions(X,labels=attributeNames)
#T = mat2transactions(Xbin,labels=attributeNamesBin)
rules = apriori(T, min_support=0.05, min_confidence=.8)
print_apriori_rules(rules)

