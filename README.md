# Logistic Regression 📈

Logistic Regression is a **supervised classification algorithm** used to predict
binary outcomes (0/1, Yes/No, Pass/Fail).

## Why Logistic Regression?
- Simple & interpretable
- Strong baseline for classification
- Used in real-world systems (fraud detection, medical diagnosis, spam filtering)

## Concepts Covered
- Sigmoid function
- Decision boundary
- Binary classification
- Confusion matrix
- Accuracy, Precision, Recall

## Implementation
1. Logistic Regression from scratch
2. Logistic Regression using Scikit-learn

> Old algorithm. Timeless relevance.
# Machine-Learning-Foundations

# K-Nearest Neighbors (KNN) Classification

This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm using Python and Scikit-learn.

## Overview
KNN is a distance-based supervised learning algorithm used for classification tasks.  
It predicts the class of a data point based on the majority class among its nearest neighbors.

## Workflow
- Loaded and explored the dataset
- Selected relevant features and target variable
- Applied train-test split
- Performed feature scaling using StandardScaler
- Trained KNN classifier
- Evaluated model using confusion matrix and accuracy score
- Compared training and testing accuracy to analyze bias–variance behavior

## Key Learnings
- Feature scaling is critical for distance-based algorithms
- Model performance depends heavily on data distribution
- Bias and variance analysis helps understand overfitting and underfitting

## Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn

## Output
- Confusion Matrix
- Accuracy Score

# Support Vector Machine (SVM) Classification

This project implements Support Vector Machine (SVM) for binary classification using Python and Scikit-learn.

## Overview
SVM is a powerful supervised learning algorithm that works by finding the optimal hyperplane that maximizes the margin between different classes.

## Workflow
- Loaded and explored the dataset
- Selected relevant features and target variable
- Applied train-test split
- Performed feature scaling using StandardScaler
- Trained SVM classifier (SVC)
- Evaluated model using confusion matrix and accuracy score
- Compared training and testing accuracy to understand bias–variance behavior

## Key Learnings
- Feature scaling is mandatory for SVM
- Margin maximization improves generalization
- Kernel-based methods enable non-linear decision boundaries

## Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn

## Output
- Confusion Matrix
- Accuracy Score

# Naive Bayes Classification (GaussianNB)

This project implements the Naive Bayes classification algorithm using Gaussian Naive Bayes from Scikit-learn.

## Overview
Naive Bayes is a probabilistic supervised learning algorithm based on Bayes’ theorem with the assumption of feature independence.

## Workflow
- Loaded and explored the dataset
- Selected relevant features and target variable
- Applied train-test split
- Performed feature scaling using StandardScaler
- Trained Gaussian Naive Bayes classifier
- Evaluated model using confusion matrix and accuracy score

## Key Learnings
- Naive Bayes is fast and computationally efficient
- Performs well even with strong independence assumptions
- Suitable for baseline and large-scale problems

## Tech Stack
- Python
- Pandas
- Scikit-learn

## Output
- Confusion Matrix
- Accuracy Score

# Decision Tree Classification

This project demonstrates the implementation of a Decision Tree classifier using Python and Scikit-learn.

## Overview
Decision Trees are supervised learning algorithms that split data into branches based on feature conditions, forming a tree-like structure that is easy to interpret.

## Workflow
- Loaded and explored the dataset
- Selected relevant features and target variable
- Applied train-test split
- Performed feature scaling
- Trained Decision Tree classifier using entropy criterion
- Evaluated model using confusion matrix and accuracy score

## Model Configuration
- Criterion: Entropy
- Max Depth: 5
- Random State: 0

## Key Learnings
- Decision Trees can model non-linear relationships effectively
- Increasing tree depth can lead to overfitting
- Limiting max depth improves generalization

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn

## Output
- Confusion Matrix
- Accuracy Score
# Random Forest Classification

This project implements the Random Forest classification algorithm using Python and Scikit-learn on the Social Network Ads dataset.

## Overview
Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive performance and reduce overfitting.

## Workflow
- Loaded and explored the dataset
- Selected relevant features (Age, Estimated Salary)
- Applied train-test split
- Performed feature scaling
- Trained Random Forest classifier using entropy criterion
- Evaluated the model using confusion matrix and accuracy score

## Model Configuration
- Number of Trees (n_estimators): 30
- Criterion: Entropy
- Max Depth: 4
- Random State: 0

## Key Learnings
- Random Forest reduces variance compared to a single Decision Tree
- Ensemble learning improves model stability and generalization
- Increasing the number of trees improves robustness but increases computation

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn

## Output
- Confusion Matrix
- [[55  3]
 [ 1 21]]
- Accuracy Score 0.95
- bias 0.921875
# XGBoost Classification

This project implements the XGBoost (Extreme Gradient Boosting) algorithm for binary classification using Python on the Social Network Ads dataset.

## Overview
XGBoost is a powerful ensemble learning algorithm based on gradient boosting that builds decision trees sequentially to correct the errors of previous models.

## Workflow
- Loaded and explored the dataset
- Selected relevant features (Age, Estimated Salary)
- Applied train-test split
- Performed feature scaling
- Trained XGBoost classifier with controlled depth and learning rate
- Evaluated model using confusion matrix and accuracy score

## Model Configuration
- Number of Trees (n_estimators): 100
- Max Depth: 4
- Learning Rate: 0.1
- Subsample: 0.8
- Column Sample by Tree: 0.8
- Evaluation Metric: Log Loss
- Random State: 0

## Key Learnings
- Gradient boosting improves performance by learning from residual errors
- Regularization helps prevent overfitting
- XGBoost provides excellent accuracy and generalization on structured data

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Output
- Confusion Matrix
- Accuracy Score
