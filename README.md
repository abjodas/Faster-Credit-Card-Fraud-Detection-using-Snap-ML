
# Credit Card Fraud Detection using Snap ML and scikit-learn

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Training](#model-training)
    - [Snap ML](#snap-ml)
    - [scikit-learn](#scikit-learn)
7. [Model Evaluation](#model-evaluation)
8. [Performance Comparison](#performance-comparison)
9. [Results](#results)
10. [Visualization](#visualization)
11. [Conclusion](#conclusion)
12. [Future Work](#future-work)
13. [References](#references)

## Introduction

Credit card fraud is a significant problem in financial sectors, leading to substantial financial losses each year. Detecting fraudulent transactions in real-time is crucial for mitigating these losses and ensuring the security of customers' financial information.

In this project, we utilize two powerful machine learning libraries, Snap ML and scikit-learn, to build and compare the performance of decision tree classifiers for detecting fraudulent credit card transactions. Snap ML is optimized for high-performance machine learning, leveraging hardware acceleration to reduce training time significantly.

## Project Overview

This project focuses on building and evaluating machine learning models to detect fraudulent transactions in a credit card dataset. We will:

1. Preprocess the data to ensure it is suitable for model training.
2. Train decision tree classifiers using Snap ML and scikit-learn.
3. Evaluate and compare the performance of these models.
4. Visualize the results to understand the models' effectiveness.

## Installation

To replicate this project, ensure you have the following libraries installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- Snap ML
- matplotlib

You can install the required libraries using pip:

\`\`\`bash
pip install pandas numpy scikit-learn snapml matplotlib
\`\`\`

## Dataset

The dataset used for this project is a publicly available credit card fraud detection dataset. It contains transactions made by credit cards in September 2013 by European cardholders. The dataset presents transactions that occurred over two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

You can download the dataset from [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Data Preprocessing

Data preprocessing is a critical step in any machine learning project. Here, we will:

1. Load the dataset.
2. Handle missing values (if any).
3. Standardize the feature set.
4. Normalize the data.

### Loading the Dataset

\`\`\`python
import pandas as pd

# Load the dataset
data = pd.read_csv('creditcard.csv')
\`\`\`

### Handling Missing Values

\`\`\`python
# Check for missing values
print(data.isnull().sum())

# Since the dataset is clean, we proceed without additional handling
\`\`\`

### Standardizing the Feature Set

We standardize the features by removing the mean and scaling to unit variance:

\`\`\`python
from sklearn.preprocessing import StandardScaler

# Standardize features
data.iloc[:, 1:30] = StandardScaler().fit_transform(data.iloc[:, 1:30])
data_matrix = data.values

# Feature matrix (excluding 'Time')
X = data_matrix[:, 1:30]

# Labels vector
y = data_matrix[:, 30]
\`\`\`

### Normalizing the Data

Normalization ensures each sample has equal weight:

\`\`\`python
from sklearn.preprocessing import normalize

# Normalize the feature matrix
X = normalize(X, norm='l1')
\`\`\`

## Model Training

### Snap ML

Snap ML is an optimized library for high-performance machine learning. It leverages hardware acceleration to significantly reduce training time.

\`\`\`python
from snapml import DecisionTreeClassifier

# Snap ML Decision Tree Classifier
snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=35, n_jobs=15)

# Train the model
snapml_dt.fit(X, y)
\`\`\`

### scikit-learn

scikit-learn is a widely used machine learning library in Python, known for its simplicity and efficiency.

\`\`\`python
from sklearn.tree import DecisionTreeClassifier

# scikit-learn Decision Tree Classifier
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

# Train the model
sklearn_dt.fit(X, y)
\`\`\`

## Model Evaluation

To evaluate the performance of our models, we will use common metrics such as accuracy, precision, recall, and F1-score.

\`\`\`python
from sklearn.metrics import classification_report

# Predict with Snap ML model
y_pred_snapml = snapml_dt.predict(X)

# Predict with scikit-learn model
y_pred_sklearn = sklearn_dt.predict(X)

# Evaluation report for Snap ML model
print("Snap ML Decision Tree Classifier Report:")
print(classification_report(y, y_pred_snapml))

# Evaluation report for scikit-learn model
print("scikit-learn Decision Tree Classifier Report:")
print(classification_report(y, y_pred_sklearn))
\`\`\`

## Performance Comparison

One of the key aspects of this project is to compare the performance of Snap ML and scikit-learn decision tree classifiers.

- Training time
- Prediction accuracy
- Resource utilization (CPU, memory)

## Results

Present the results of your model evaluations, including metrics and any interesting findings.

## Visualization

Visualizations help in understanding the data and the performance of the models better. Use matplotlib to create plots such as:

- Confusion matrices
- ROC curves
- Precision-Recall curves

## Conclusion

Summarize the findings of your project, highlighting the strengths and weaknesses of using Snap ML compared to scikit-learn.

## Future Work

Discuss potential improvements and future directions for this project, such as:

- Exploring other machine learning models.
- Using more advanced techniques for handling imbalanced data.
- Leveraging other hardware-accelerated libraries.

## References

List any references, including academic papers, articles, and documentation used in the project.
