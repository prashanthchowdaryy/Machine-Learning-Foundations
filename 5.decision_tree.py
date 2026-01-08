# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:12:43 2026

@author: Prashanth
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 18:56:08 2026

@author: Prashanth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importinng the data

dataset=pd.read_csv(r"C:\Users\Prashanth\Desktop\Naresh_it\MachineLearning\6th - Navie Bayes\6th - Navie Bayes\Social_Network_Ads.csv")
x=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

# splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# using feature scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)
classifier.fit(x_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=4,n_estimators=30, criterion="entropy", random_state=0)
classifier.fit(x_train, y_train)
# predicting the test set results

y_pred=classifier.predict(x_test)

# making the confusion matrix

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

bias=classifier.score(x_train,y_train)
bias

