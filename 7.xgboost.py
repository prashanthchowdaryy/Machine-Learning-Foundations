# -*- coding: utf-8 -*-
"""
XGBoost Classification
@author: Prashanth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset=pd.read_csv(r"C:\Users\Prashanth\Desktop\Naresh_it\MachineLearning\Data\Churn_Modelling.csv")
x=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])

# till here we have complete in class

# Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=0
)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train XGBoost model
from xgboost import XGBClassifier

classifier = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=0
)

classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

# Evaluation
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", ac)

# Bias (training accuracy)
bias = classifier.score(X_train, y_train)
print("Training Accuracy (Bias):", bias)
