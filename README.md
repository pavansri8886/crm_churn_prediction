# CRM Churn Prediction — ML Project

> **MSc Data Management | ML Deployment Course | Portfolio Project**
> Predicting customer churn using CRM behavioral data, with full MLOps deployment.

---

## 🎯 Business Problem

Acquiring a new customer costs **5–7× more** than retaining an existing one.
This project builds a production-ready ML system that:
1. Identifies customers at risk of churning (with probability scores)
2. Recommends automated CRM actions (HubSpot / Salesforce integration)
3. Deploys as a REST API consumed by CRM dashboards

**Real-world relevance:** This is core functionality in Salesforce Einstein, HubSpot's AI, and Ogury's Sales Effectiveness tooling.

---

## 🏗️ Project Architecture

```
crm_churn_prediction/
│
├── config.py                  ← Central config (all params in one place)
├── run_pipeline.py            ← ONE command to run everything
├── requirements.txt
│
├── src/
│   ├── data_loader.py         ← Load real or synthetic Telco CRM data
│   ├── preprocessing.py       ← Type fixing, imputation, target encoding
│   ├── feature_engineering.py ← CRM domain features + sklearn ColumnTransformer
│   ├── train.py               ← MLflow-tracked multi-model training
│   └── evaluate.py            ← Metrics, ROC curves, SHAP explainability
│
├── api/
│   └── main.py                ← FastAPI REST endpoint (single + batch predict)
│
├── dashboard/
│   └── app.py                 ← Streamlit CRM dashboard
│
├── tests/
│   └── test_pipeline.py       ← pytest unit & integration tests
│
├── data/                      ← Raw & processed datasets
├── models/                    ← Serialized model pipelines (joblib)
└── logs/plots/                ← ROC curves, confusion matrix, SHAP plots
```

---

## 🔬 ML Pipeline

### 1. Data
- **Dataset:** IBM Telco Customer Churn (~7,000 customers, 20 features)
- **Source:** [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Auto-fallback:** Generates realistic synthetic data if CSV not found

### 2. Preprocessing
| Step | Technique |
|------|-----------|
| Type coercion | `TotalCharges` string → float |
| Missing values | Median (numerical), Mode (categorical) |
| Outlier capping | IQR method |
| Target encoding | Churn: Yes→1, No→0 |
| Stratified split | 70% train / 10% val / 20% test |

### 3. Feature Engineering (CRM Domain)
Custom business-logic features that encode domain knowledge:

| Feature | Business Logic |
|---------|---------------|
| `ChargesPerTenureMonth` | Revenue density — high early charges signal risk |
| `IsNewCustomer` | tenure ≤ 6 months (highest churn window) |
| `IsLongTerm` | tenure ≥ 24 months (stickier customers) |
| `ServiceBundleScore` | # of services subscribed (higher = stickier) |
| `ContractRiskScore` | Month-to-month=3, 1yr=2, 2yr=1 |
| `TotalChargesGap` | Expected vs actual charges (billing anomalies) |
| `HighRiskCombo` | Month-to-month + Fiber + No security (triple risk) |

### 4. Models Trained
| Model | Why Included |
|-------|-------------|
| Logistic Regression | Interpretable baseline, fast |
| Random Forest | Robust ensemble, handles non-linearity |
| **XGBoost** | Best performer (gradient boosting) |
| LightGBM | Fast alternative to XGBoost |

### 5. Evaluation
- **Primary metric:** ROC-AUC (handles class imbalance)
- **Secondary:** F1, Precision, Recall, PR-AUC
- **Explainability:** SHAP beeswarm plots (feature importance)
- **Optimal threshold:** Maximized on validation set (not default 0.5)

### 6. Deployment
- **MLflow:** Experiment tracking, model registry
- **FastAPI:** REST API with Pydantic validation
- **Streamlit:** Interactive CRM dashboard

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Download real dataset
```bash
# Download from Kaggle and place here:
# data/telco_churn.csv
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#
# If you skip this, synthetic data is auto-generated.
```

### 3. Run the full pipeline
```bash
python run_pipeline.py
```

This runs all 7 steps: data → clean → split → train 4 models → evaluate → SHAP → save.

### 4. Launch the API
```bash
uvicorn api.main:app --reload --port 8000
```
Open: http://localhost:8000/docs (interactive Swagger UI)

### 5. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

### 6. View MLflow experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open: http://localhost:5000

### 7. Run tests
```bash
pytest tests/ -v
```

---

## 📡 API Reference

### `POST /predict` — Single customer
```json
{
  "tenure": 12,
  "MonthlyCharges": 65.50,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  ...
}
```

**Response:**
```json
{
  "churn_probability": 0.7823,
  "churn_prediction": 1,
  "risk_level": "high",
  "recommended_action": "URGENT: Escalate to retention team immediately.",
  "model_version": "xgboost-v1.0",
  "timestamp": "2026-03-10T14:22:01"
}
```

### `POST /predict/batch` — Bulk scoring
Send a list of customer objects. Returns individual scores + aggregate summary.

---

## 📊 Expected Results

| Model | Val ROC-AUC | Test ROC-AUC | Test F1 |
|-------|-------------|--------------|---------|
| Logistic Regression | ~0.83 | ~0.82 | ~0.60 |
| Random Forest | ~0.86 | ~0.85 | ~0.63 |
| **XGBoost** | **~0.88** | **~0.87** | **~0.66** |
| LightGBM | ~0.87 | ~0.86 | ~0.65 |

*Results will vary slightly with synthetic data vs. real Telco dataset.*

---

## 📚 Dataset Citation

IBM Telco Customer Churn Dataset  
Available at: https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
License: Community Data License Agreement

---

## 🛠️ Tech Stack

`Python 3.10+` · `scikit-learn` · `XGBoost` · `LightGBM` · `MLflow` · `SHAP` · `FastAPI` · `Streamlit` · `Plotly` · `pandas` · `pytest`
