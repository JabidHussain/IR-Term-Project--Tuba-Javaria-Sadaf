                    # Applying Naive bayes on Dbworld dataset 

#import libraries

import numpy as np
import pandas as pd
import urllib
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

# Setting path of dataset
address='C:/Users/jabid/Desktop/dbworld_bodies.csv'
dbworld = pd.read_csv(address).values
y=dbworld[:,-1]               #dividing in dependent independent dataset
x=dbworld[:,0:4702]


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)   #trainig and testing of data on 30% 70%
 

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)   #trainig and testing of data on 30% 70%
BernNB = BernoulliNB(binarize=True)    #Bernouli Classifier
BernNB.fit(xtrain, ytrain)
print (BernNB)
y_expect= ytest
y_pred = BernNB.predict(xtest)
print accuracy_score(y_expect, y_pred)     # prinintg accuracy

MultiNB = MultinomialNB()                  # multinomial classifier
MultiNB.fit(xtrain, ytrain)
print (MultiNB)
y_pred = MultiNB.predict(xtest)
print accuracy_score(y_expect, y_pred)

GausNB = GaussianNB()                      # Gaussian classifier
GausNB.fit(xtrain, ytrain)
print (GausNB)
y_pred = GausNB.predict(xtest)
print accuracy_score(y_expect, y_pred)