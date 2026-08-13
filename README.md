# 📊 Customer Churn Prediction + Explainable AI

An AI/ML powered Customer Churn Prediction system that predicts whether a customer is likely to leave a service and explains the model's prediction using Explainable AI (SHAP).

## 🚀 Live Demo

👉 https://tushar-customer-churn.streamlit.app

## 💻 GitHub Repository

👉 https://github.com/tusharsharma1452/customer-churn-ai

---

## 📌 Project Overview

Customer churn is a major problem for telecom, banking, SaaS and e-commerce companies.

This project uses machine learning to predict the probability of a customer leaving a service.

The system also uses SHAP (SHapley Additive exPlanations) to explain why the model made a particular prediction.

---

## 🎯 Objectives

- Predict customer churn
- Calculate churn probability
- Classify customers into risk levels
- Compare multiple machine learning models
- Explain individual predictions using SHAP
- Provide an interactive web application
- Deploy the model online using Streamlit

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Streamlit
- Joblib

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
One-Hot Encoding
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Comparison
   ↓
XGBoost Model
   ↓
SHAP Explainability
   ↓
Streamlit Application
   ↓
Deployment
```

---

## 🤖 Machine Learning Models

Three models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.70% | 65.84% | 56.68% | 60.92% | 84.16% |
| XGBoost | 80.06% | 65.87% | 51.60% | 57.87% | 84.20% |
| Random Forest | 78.92% | 63.05% | 49.73% | 55.61% | 82.46% |

XGBoost achieved the highest ROC-AUC among the three models, while Logistic Regression achieved the highest accuracy.

---

## 🧠 Explainable AI with SHAP

SHAP is used to understand how individual features influence the model's prediction.

The application provides a SHAP waterfall plot that shows:

- Features pushing the prediction higher
- Features pushing the prediction lower
- The contribution of individual features
- The overall model output

This makes the machine learning prediction easier to interpret.

---

## 🌐 Streamlit Application

The Streamlit application allows users to enter customer information such as:

- Gender
- Senior Citizen status
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

The application returns:

- Churn Probability
- Risk Level
- Prediction
- Customer Summary
- SHAP-based explanation

---

## 📈 Example Prediction

For a sample customer:

```text
Churn Probability: 25.45%
Prediction: Customer is likely to stay
Risk Level: Low Risk
```

The same prediction was verified in both the notebook and deployed Streamlit application.

---

## 📂 Project Structure

```text
customer-churn-ai/
│
├── app.py
├── churn_model.pkl
├── feature_columns.pkl
├── categorical_columns.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/tusharsharma1452/customer-churn-ai.git
```

Move into the project directory:

```bash
cd customer-churn-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🔮 Future Improvements

- Add customer retention recommendations
- Add batch prediction using CSV upload
- Add model monitoring
- Improve UI and visual analytics
- Add more advanced model tuning
- Add customer segmentation
- Add database integration

---

## 👨‍💻 Author

**Tushar Sharma**

B.Tech Computer Science Engineering

---

## ⭐ Project Highlights

- End-to-end machine learning project
- Multiple model comparison
- XGBoost
- Explainable AI using SHAP
- Interactive Streamlit application
- GitHub version control
- Cloud deployment