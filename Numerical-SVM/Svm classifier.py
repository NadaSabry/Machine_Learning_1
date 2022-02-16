# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 18:12:23 2021

@author: Eman Ashraf
"""

import numpy as np
#library that contains mathematical tools

import matplotlib.pyplot as plt
#library that help us plot nice charts

import pandas as pd
#best library to import and manage datasets 
import tensorflow as tf
import seaborn as sb
#importing the dataset
dataset = pd.read_csv('TravelInsurancePrediction.csv')

X = dataset.iloc[: , 1:9].values 
# our input(features) equals all rows and columns from 1 to 9

y = dataset.iloc[: , 9].values
# our output equals all rows and the last column only

from sklearn.preprocessing import LabelEncoder
#encoding categorial data -> class from sklearn
 # we fit to the second column
labelEncoder_X1 = LabelEncoder()
X[:, 1] = labelEncoder_X1.fit_transform(X[:, 1])

labelEncoder_X2 = LabelEncoder()
X[:, 2] = labelEncoder_X2.fit_transform(X[:, 2])
labelEncoder_X3 = LabelEncoder()
X[:, 6] = labelEncoder_X3.fit_transform(X[:, 6])
labelEncoder_X4 = LabelEncoder() 
X[:, 7] = labelEncoder_X4.fit_transform(X[:, 7])

#splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state= 0)
#splitting the training dataset into training and validating = 0.34*0.3 = 0.102 
#X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.34, random_state = 0 )

#features scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() #object from the class to scale x matrix
X = sc_X.fit_transform(X)
X_train = sc_X.fit_transform(X_train)
#fit the object and then transform the training set
X_test = sc_X.transform(X_test)

#classifier 
from sklearn.svm import SVC
classifier = SVC(kernel=('poly'), random_state = 0, probability=(1))
classifier.fit(X_train, y_train)
#prediction
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred)
s = accuracy_score(y_test, y_pred)

plot_confusion_matrix(classifier, X_test, y_test)
plt.show()

#ROC curve 

from sklearn.metrics import roc_curve, auc, roc_auc_score
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred)
    roc_auc[i] = auc(fpr[i], tpr[i])

print(roc_auc_score(y_test, y_pred))
plt.figure()
plt.plot(fpr[1], tpr[1])
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


#learning curve
from yellowbrick.model_selection import learning_curve
print(learning_curve(classifier, X_train, y_train, cv=10, scoring='accuracy')) 

