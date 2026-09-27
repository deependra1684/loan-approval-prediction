# Loan Approval Prediction

# Project Overview

This project focuses on predicting whether a loan application will be **Approved or Rejected** using machine learning classification algorithms.

The project also focuses on handling **class imbalance**, since loan approval data contains significantly more rejected applications than approved applications.

The goal is to build and evaluate machine learning models and understand how different approaches affect classification performance.

---

# Problem Statement

Financial institutions receive a large number of loan applications. Predicting loan approval based on applicant and loan-related characteristics can help in understanding the factors associated with approval decisions.

This project uses machine learning classification techniques to predict the loan status.

# Dataset

The dataset contains **100,000 loan application records** and **14 columns**.

The target variable is:

**LoanStatus**

- `Approved`
- `Rejected`

The dataset contains applicant information, financial information, credit-related features, and loan characteristics.

# Class Distribution

- Rejected: **88.587%**
- Approved: **11.413%**

This shows a significant **class imbalance** in the target variable.

---

# Exploratory Data Analysis

The following areas were explored:

- Dataset shape and structure
- Missing values
- Duplicate records
- Target variable distribution
- Numerical feature statistics
- Outlier analysis
- Feature correlation
- Relationship between features and loan approval

No missing values were found in the dataset.

---

#  Data Preprocessing

The following preprocessing steps were performed:

- Separation of features and target
- Encoding of the target variable
- Train-test split
- Numerical feature processing
- One-hot encoding of categorical features
- Feature transformation using `ColumnTransformer`
- Handling unknown categories using `handle_unknown='ignore'`

The processed dataset contained **24 features** after encoding.

---

#  Handling Class Imbalance

Since the dataset contains substantially more rejected applications than approved applications, accuracy alone is not sufficient for evaluating the models.

Different approaches were explored, including:

- Class weights
- `scale_pos_weight` in XGBoost
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion matrix

Special attention was given to the **Approved** class because correctly identifying approved applications is important for understanding model performance on the minority class.

---

#  Machine Learning Models

The following classification algorithms were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Model performance was compared using classification metrics rather than relying only on accuracy.

---

# XGBoost & Hyperparameter Tuning

XGBoost was further explored using hyperparameter tuning.

Important parameters included:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `scale_pos_weight`

RandomizedSearchCV was used for hyperparameter search.

The selected XGBoost configuration included:

```text
n_estimators = 200
max_depth = 5
learning_rate = 0.1
scale_pos_weight = 2
