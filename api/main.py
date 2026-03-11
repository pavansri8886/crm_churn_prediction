"""
api/main.py — FastAPI REST API for CRM Churn Prediction.

Endpoints:
  GET  /health        — health check
  POST /predict       — single customer prediction
  POST /predict/batch — batch prediction (list of customers)
  GET  /model/info    — model metadata

Run with:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from loguru import logger

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import BEST_MODEL_PATH, CHURN_RISK_THRESHOLDS


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CRM Churn Prediction API",
    description=(
        "Predict customer churn probability using an XGBoost model "
        "trained on CRM behavioral data. Integrates with HubSpot / Salesforce."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Load model at startup ────────────────────────────────────────────────────

pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    if BEST_MODEL_PATH.exists():
        pipeline = joblib.load(BEST_MODEL_PATH)
        logger.info(f"Model loaded from {BEST_MODEL_PATH}")
    else:
        logger.warning(
            f"Model not found at {BEST_MODEL_PATH}. "
            "Run src/train.py first to train and save models."
        )


# ─── Schemas ──────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """
    Input schema matching the Telco CRM dataset columns.
    Maps directly to HubSpot / Salesforce CRM field names.
    """
    tenure:            int   = Field(..., ge=0, le=120,  example=12,    description="Months as customer")
    MonthlyCharges:    float = Field(..., ge=0, le=500,  example=65.50, description="Monthly bill amount (€)")
    TotalCharges:      Optional[float] = Field(None,     example=786.0, description="Total amount charged")
    gender:            str   = Field(..., example="Male")
    Partner:           str   = Field(..., example="Yes")
    Dependents:        str   = Field(..., example="No")
    PhoneService:      str   = Field(..., example="Yes")
    MultipleLines:     str   = Field(..., example="No")
    InternetService:   str   = Field(..., example="Fiber optic")
    OnlineSecurity:    str   = Field(..., example="No")
    OnlineBackup:      str   = Field(..., example="No")
    DeviceProtection:  str   = Field(..., example="No")
    TechSupport:       str   = Field(..., example="No")
    StreamingTV:       str   = Field(..., example="Yes")
    StreamingMovies:   str   = Field(..., example="Yes")
    Contract:          str   = Field(..., example="Month-to-month")
    PaperlessBilling:  str   = Field(..., example="Yes")
    PaymentMethod:     str   = Field(..., example="Electronic check")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.0,
                "gender": "Male",
                "Partner": "No",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Probability of churn (0–1)")
    churn_prediction:  int   = Field(..., description="1 = likely to churn, 0 = likely to stay")
    risk_level:        str   = Field(..., description="low / medium / high")
    recommended_action: str  = Field(..., description="CRM action recommendation")
    model_version:     str
    timestamp:         str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers:    int
    high_risk_count:    int
    medium_risk_count:  int
    low_risk_count:     int


# ─── Helper functions ─────────────────────────────────────────────────────────

def get_risk_level(prob: float) -> str:
    for level, (lo, hi) in CHURN_RISK_THRESHOLDS.items():
        if lo <= prob < hi:
            return level
    return "high"


def get_recommended_action(risk_level: str, prob: float) -> str:
    actions = {
        "low":    "No immediate action needed. Schedule quarterly check-in.",
        "medium": "Send retention offer email. Assign to Customer Success team.",
        "high":   f"URGENT: {prob:.0%} churn risk. Escalate to retention team immediately. Offer contract upgrade discount."
    }
    return actions[risk_level]


def predict_single(customer: CustomerFeatures) -> PredictionResponse:
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run train.py first."
        )

    df = pd.DataFrame([customer.dict()])
    prob  = float(pipeline.predict_proba(df)[0, 1])
    pred  = int(prob >= 0.5)
    risk  = get_risk_level(prob)
    action = get_recommended_action(risk, prob)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_level=risk,
        recommended_action=action,
        model_version="xgboost-v1.0",
        timestamp=datetime.utcnow().isoformat()
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status":       "healthy",
        "model_loaded": pipeline is not None,
        "timestamp":    datetime.utcnow().isoformat()
    }


@app.get("/model/info", tags=["System"])
def model_info():
    return {
        "model_name":    "XGBoost CRM Churn Predictor",
        "version":       "1.0.0",
        "framework":     "scikit-learn Pipeline + XGBoost",
        "features":      18,
        "target":        "Churn (binary)",
        "training_data": "IBM Telco Customer Churn Dataset",
        "crm_compatible": ["HubSpot", "Salesforce", "Custom CRM"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.
    Returns risk level and recommended CRM action.
    """
    logger.info(f"Single prediction request received")
    return predict_single(customer)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_churn_batch(customers: List[CustomerFeatures]):
    """
    Predict churn for a batch of customers.
    Returns individual predictions + aggregate risk summary.
    """
    logger.info(f"Batch prediction: {len(customers)} customers")

    predictions = [predict_single(c) for c in customers]

    risk_counts = {
        "high":   sum(1 for p in predictions if p.risk_level == "high"),
        "medium": sum(1 for p in predictions if p.risk_level == "medium"),
        "low":    sum(1 for p in predictions if p.risk_level == "low"),
    }

    return BatchPredictionResponse(
        predictions=predictions,
        total_customers=len(predictions),
        high_risk_count=risk_counts["high"],
        medium_risk_count=risk_counts["medium"],
        low_risk_count=risk_counts["low"]
    )
