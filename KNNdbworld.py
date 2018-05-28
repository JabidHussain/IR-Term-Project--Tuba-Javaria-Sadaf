                          #KNN classifier on Database Dbworld



#import libraries
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from pylab import rcParams
import urllib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors 
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
address='C:/Users/jabid/Desktop/dbworld_bodies.csv'      #path of csv file
dbworld = pd.read_csv(address).values
y=dbworld[:,-1]                                          #label of testing class
x=dbworld[:,0:4702]                                      #label of training class




xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)   # training the data

clf = neighbors.KNeighborsClassifier()   #calling KNN classifier
clf.fit(xtrain,ytrain)
print(clf)
y_expect= ytest
y_pred = clf.predict(xtest)
print(metrics.classification_report(y_expect, y_pred))      #print metrics

