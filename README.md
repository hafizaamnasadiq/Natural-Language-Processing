# Natural-Language-Processing

# Confusion Metrics Calculator 📊

This project implements a simple **Confusion Matrix-based evaluation tool** from scratch in Python.  
It computes key classification metrics such as Accuracy, Precision, Recall, and F1-score.

---

## 📌 Features

- Builds confusion matrix labels (TP, TN, FP, FN)
- Computes:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Handles edge cases (division by zero)
- Includes multiple test cases for validation

- # Naive Bayes Python Implementation
  Kaggle link for dataset : https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download
-Developed a Multinomial Naive Bayes classifier using only standard Python, adhering to the "no machine learning libraries" constraint.  
-IMDB Sentiment Task: Adapted the code to classify movie reviews by mapping sentiments to binary labels for statistical analysis. 
-Laplace Smoothing: "zero probability problem" for words not encountered during the training phase. 
-Numerical Stability: Utilized log-probability summation to prevent floating-point underflow in long text documents.  
-Integrated Evaluation: Reused a custom ConfusionMetrics class to calculate accuracy, precision, recall, and F1 scores.

