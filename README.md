# Customer Churn Prediction using Machine Learning

A machine learning project that predicts whether a telecom customer is likely to churn (leave the company's service), using the Telco Customer Churn dataset. The project covers the full data science workflow — data cleaning, exploratory data analysis, feature engineering, model building, model tuning, and interpretability — and finishes with a saved, ready-to-use churn prediction model.

## Project Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses such as telecom companies, banks, insurance providers, and streaming platforms. Losing existing customers directly impacts revenue and growth, and retaining a customer is generally far cheaper than acquiring a new one.

This project builds and compares several classification models to predict customer churn based on demographic information, account details, and service usage patterns, then identifies the key factors driving churn so a business can act on them.

## Dataset

- **Name:** Telco Customer Churn
- **Source:** Kaggle ([link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))
- **Records:** 7,043 customers
- **Features:** 21 (demographics, account info, subscribed services, billing details)
- **Target variable:** `Churn` (Yes / No)

## Project Workflow

1. **Data Loading & Inspection** — load the dataset and examine its structure, data types, and summary statistics
2. **Data Cleaning** — handle hidden missing values in `TotalCharges`, drop unnecessary identifier columns
3. **Exploratory Data Analysis (EDA)** — analyze churn against gender, senior citizen status, partner/dependents, internet service, online security, tech support, contract type, payment method, monthly charges, and tenure
4. **Feature Encoding & Train-Test Split** — one-hot encode categorical variables, split data 80:20 with stratified sampling
5. **Model Building** — train and evaluate four baseline models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - K-Nearest Neighbors (KNN)
6. **Model Comparison & Feature Importance** — compare all models on Accuracy, Precision, Recall, F1-Score, and ROC-AUC; identify top churn-driving features
7. **Advanced Improvements:**
   - Handling class imbalance (`class_weight='balanced'`)
   - 5-fold cross-validation for result stability
   - Hyperparameter tuning (GridSearchCV) on Random Forest
   - Gradient boosting model (XGBoost)
   - ROC curve comparison across all models
   - Decision threshold tuning & precision-recall trade-off analysis
   - Model interpretability using SHAP
   - Saving the final model with `joblib`

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Logistic Regression** | **80.31%** | **0.836** |
| Random Forest (tuned) | 79.60% | 0.838 |
| XGBoost | 79.46% | 0.838 |
| Random Forest (default) | 78.96% | 0.816 |
| K-Nearest Neighbors | 75.34% | 0.767 |
| Decision Tree | 71.86% | 0.637 |

**Logistic Regression** was selected as the final model — it achieved the best overall balance of accuracy, recall, and ROC-AUC, and its 5-fold cross-validated performance (mean ROC-AUC 0.845) confirmed the result was stable, not a product of one lucky train-test split.

### Key Churn Drivers

Based on feature importance and SHAP analysis, the strongest predictors of churn are:
- **Tenure** (newer customers churn far more often)
- **Contract type** (month-to-month contracts churn much more than one/two-year contracts)
- **Internet service type** (Fiber optic customers churn the most)
- **Payment method** (Electronic check users churn the most)
- **Online security & tech support** (customers without these services churn more)
- **Monthly charges** (higher charges are associated with higher churn)

## Business Recommendations

- Encourage customers to switch from month-to-month contracts to long-term contracts
- Improve technical support and online security services to reduce churn
- Offer loyalty rewards to long-term customers
- Monitor and address customers with high monthly charges
- Design targeted retention campaigns for customers flagged as high-risk by the model

## Tech Stack

- **Language:** Python
- **Data Manipulation:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Model Interpretability:** SHAP
- **Model Persistence:** joblib

## Project Structure

```
├── Customer_Churn_Prediction.ipynb   # Main notebook (end-to-end workflow)
├── data/
│   └── customer_churn.csv            # Telco Customer Churn dataset
├── churn_model_logistic_regression.pkl  # Saved final model (generated after running the notebook)
└── README.md
```

## How to Run

1. Clone or download this repository
2. Install the required dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib
   ```
3. Make sure `customer_churn.csv` is placed inside a `data/` folder in the project directory
4. Open `Customer_Churn_Prediction.ipynb` in Jupyter Notebook / JupyterLab and run all cells

## Using the Saved Model

```python
import joblib

model = joblib.load('churn_model_logistic_regression.pkl')

# X_new should be a one-hot encoded DataFrame with the same columns used in training
prediction = model.predict(X_new)
probability = model.predict_proba(X_new)[:, 1]
```

## 👩‍💻 Author

**Memoona Abbas**

Bachelor of Science (BS) in Computer Science

Aspiring AI Engineer with a strong interest in Artificial Intelligence, Machine Learning, Data Science, Large Language Models (LLMs), and Generative AI. Passionate about building practical AI solutions and continuously expanding technical expertise through real-world projects.