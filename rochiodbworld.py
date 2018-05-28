                             #Applying Rocchio classifier on Dbworld database

    
#import libraries
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from pylab import rcParams
import urllib
import sklearn
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import neighbors 
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics

#loading dataset
address='C:/Users/jabid/Desktop/dbworld_bodies.csv'
dbworld = pd.read_csv(address).values

#definig dependent independent classes
y=dbworld[:,-1]
x=dbworld[:,0:4702]

#training data on 70% 30%dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3) 


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0) 

#applying rocchio

clf = NearestCentroid()
clf.fit(xtrain,ytrain)
NearestCentroid(metric='euclidean', shrink_threshold=None)
print(clf)


y_expect= ytest
y_pred = clf.predict(xtest)
print(metrics.classification_report(y_expect, y_pred))