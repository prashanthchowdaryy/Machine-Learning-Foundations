# Logistic Regression (Binary Classification)

This project demonstrates the implementation of Logistic Regression for binary classification using Python and Scikit-learn.

## Overview
Logistic Regression is a fundamental supervised learning algorithm used to model the probability of a binary outcome using the sigmoid function.

## Workflow
- Loaded and explored the dataset
- Selected relevant features and target variable
- Applied train-test split
- Performed feature scaling using StandardScaler
- Trained Logistic Regression classifier
- Evaluated model using confusion matrix, accuracy score, and classification report

## Key Learnings
- Logistic Regression is sensitive to feature scaling
- Confusion matrix provides deeper insight beyond accuracy
- Classification report helps evaluate precision, recall, and F1-score

## Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn

## Output
- Confusion Matrix
- Accuracy Score
- Classification Report


# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 10:01:19 2025

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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=71)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)





