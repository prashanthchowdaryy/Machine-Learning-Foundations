# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 10:13:42 2026

@author: Prashanth
"""
import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
# Load the dataset
dataset = pd.read_csv(r"C:\Users\Prashanth\Desktop\Naresh_it\MachineLearning\Data\logit classification.csv")
dataset.head()

# Split the data into independent and dependent variables
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifirer=KNeighborsClassifier()
classifirer.fit(x_train,y_train)


y_pred=classifirer.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)

bias=classifirer.score(x_train,y_train)
print(bias)

variance=classifirer.score(x_test,y_test)
print(variance)
