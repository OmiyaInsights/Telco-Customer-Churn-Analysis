# Telco-Customer-Churn-Analysis
Predicting telecom customer churn using Decision Tree and Logistic Regression, with EDA and feature importance analysis to identify key churn drivers.
# Telco Customer Churn Prediction

Predicting which telecom customers are about to churn, and explaining *why*, using a Decision Tree and a Logistic Regression model.

Customer churn prediction is a problem real companies in telecom, banking, and SaaS pay to solve — keeping an existing customer is far cheaper than acquiring a new one. This project builds an end-to-end, interpretable churn model on the classic Telco Customer Churn dataset.

## Overview

- **Dataset:** 7,043 customers, 21 features (demographics, subscribed services, account/billing info, churn label)
- **Models:** Decision Tree Classifier & Logistic Regression (scikit-learn), compared head-to-head
- **Focus:** interpretability — the goal isn't just accuracy, it's being able to tell a non-technical manager *which levers to pull*

## What's inside

1. **EDA** — churn rate by contract type, internet service, tenure, monthly charges, tech support, and payment method
2. **Data cleaning** — fixes the `TotalCharges` column (blank strings on zero-tenure customers), drops the ID column, encodes the target
3. **Categorical encoding** — one-hot encoding via `pd.get_dummies`
4. **Class imbalance** — the dataset is ~73.5% / 26.5% (stay/churn). Handled with `class_weight="balanced"` on both models; noted as a partial fix, with SMOTE/resampling flagged as a next step
5. **Modeling** — Decision Tree (`max_depth=5`) and Logistic Regression, both evaluated on a stratified 80/20 split
6. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC curve comparison (accuracy alone is misleading on imbalanced data, so recall/ROC-AUC get the most weight)
7. **Feature importance** — top churn drivers via `.feature_importances_`, grouped back to their original columns
8. **Business summary** — written for a non-technical audience
9. **Limitations & next steps**

## Key results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.735 | 0.500 | 0.808 | 0.618 | 0.831 |
| Logistic Regression | 0.740 | 0.507 | 0.786 | 0.616 | 0.841 |

**Top 3 churn drivers (Decision Tree feature importance):**
1. **Contract type** — month-to-month customers churn at ~43%, vs ~11% (1-year) and ~3% (2-year)
2. **Internet service** — fiber optic customers churn at ~42%, vs ~19% for DSL
3. **Tenure** — churn risk drops steadily the longer a customer stays

## Repo structure

```
.
├── telco_churn_analysis.ipynb   # full analysis notebook (run top to bottom)
├── telco_churn_analysis.html    # static export, viewable without Jupyter
├── telco_churn.csv              # dataset
└── README.md
```

## Running it locally

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook telco_churn_analysis.ipynb
```

## Tech stack

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

## Limitations & next steps

- Class imbalance was only partially addressed (`class_weight="balanced"`, no resampling like SMOTE yet)
- Models are untuned — a hyperparameter search would likely improve both
- A Random Forest / Gradient Boosting model would likely beat a single Decision Tree on raw metrics, at some cost to interpretability
- The 0.5 decision threshold wasn't tuned for the business cost of a missed churner vs. a false alarm

## Dataset

Telco Customer Churn dataset (commonly distributed via Kaggle / IBM sample datasets).
