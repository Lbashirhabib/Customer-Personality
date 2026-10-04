from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import pickle
import pandas as pd
import numpy as np
import uvicorn

# ── LOAD MODEL ──────────────────────────────────────
print("Loading customer personality model...")
try:
    with open("../models/customer_model.pkl", "rb") as f:
        artifacts      = pickle.load(f)
    model          = artifacts["model"]
    imputer        = artifacts["imputer"]
    encoders       = artifacts["encoders"]
    cluster_names  = artifacts["cluster_names"]
    features       = artifacts["features"]
    print("  Model loaded successfully.")
except Exception as e:
    print(f"  Load failed: {e}")
    model = None

app = FastAPI(
    title="Customer Personality Prediction API",
    description="Predicts customer segment using XGBoost",
    version="1.0.0"
)

# ── REQUEST SCHEMA ───────────────────────────────────
class Customer(BaseModel):
    Income:              float
    Recency:             int
    MntWines:            float
    MntFruits:           float
    MntMeatProducts:     float
    MntFishProducts:     float
    MntSweetProducts:    float
    MntGoldProds:        float
    NumWebPurchases:     int
    NumCatalogPurchases: int
    NumStorePurchases:   int
    NumDealsPurchases:   int
    NumWebVisitsMonth:   int
    AcceptedCmp1:        Optional[int] = 0
    AcceptedCmp2:        Optional[int] = 0
    AcceptedCmp3:        Optional[int] = 0
    AcceptedCmp4:        Optional[int] = 0
    AcceptedCmp5:        Optional[int] = 0
    Response:            Optional[int] = 0
    Kidhome:             Optional[int] = 0
    Teenhome:            Optional[int] = 0
    Age:                 Optional[int] = 40
    Education:           Optional[str] = "Graduation"
    Marital_Status:      Optional[str] = "Married"
    Customer_Days:       Optional[int] = 500
    Complain:            Optional[int] = 0


def prepare(customer: Customer) -> pd.DataFrame:
    raw = customer.dict()
    df  = pd.DataFrame([raw])

    # Engineer same features as notebook
    spend_cols    = ['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases',
                     'NumStorePurchases', 'NumDealsPurchases']
    camp_cols     = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                     'AcceptedCmp4', 'AcceptedCmp5']

    df['TotalSpend']         = df[spend_cols].sum(axis=1)
    df['TotalPurchases']     = df[purchase_cols].sum(axis=1)
    df['SpendPerPurchase']   = (df['TotalSpend'] /
                                (df['TotalPurchases'] + 1)).round(2)
    df['TotalChildren']      = df['Kidhome'] + df['Teenhome']
    df['CampaignAcceptRate'] = df[camp_cols].sum(axis=1)
    df['IsParent']           = (df['TotalChildren'] > 0).astype(int)
    df['IncomePerHead']      = (df['Income'] /
                                (df['TotalChildren'] + 1)).round(2)

    # Encode categoricals
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    # Align to training features
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]

    return pd.DataFrame(
        imputer.transform(df), columns=features)


SEGMENT_INFO = {
    'High Value':    'High income, high spend. Target with premium offers and loyalty rewards.',
    'Potential':     'Medium income, moderate spend. Target with growth campaigns and bundle deals.',
    'At Risk':       'Previously active but declining. Target with win-back campaigns and discounts.',
    'Low Engagement':'Low spend, rarely responds. Target with budget offers and basic communication.',
}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(503, "Model not loaded.")
    try:
        X          = prepare(customer)
        cluster_id = int(model.predict(X)[0])
        probas     = model.predict_proba(X)[0]
        segment    = cluster_names[cluster_id]

        return {
            "cluster_id":        cluster_id,
            "segment":           segment,
            "confidence":        round(float(probas[cluster_id]), 4),
            "marketing_action":  SEGMENT_INFO.get(segment, ""),
            "all_probabilities": {
                cluster_names[i]: round(float(p), 4)
                for i, p in enumerate(probas)
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.get("/segments")
def segments():
    return {"segments": SEGMENT_INFO}


@app.get("/stats")
def stats():
    return {
        "model":    "XGBoost Multiclass Classifier",
        "segments": list(cluster_names.values()),
        "features": len(features)
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Personality API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px;
               margin: 60px auto; padding: 20px; background: #f5f5f5; }
        h1   { color: #1a237e; }
        .box { background: white; padding: 20px; border-radius: 10px;
               margin: 16px 0; border-left: 4px solid #1a237e; }
        code { background: #eef2ff; padding: 2px 6px; border-radius: 4px; }
        a    { color: #1a237e; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Customer Personality Prediction API</h1>
    <p>Predicts which customer segment a person belongs to
       using XGBoost trained on marketing campaign data.</p>
    <div class="box">
        <h3>Endpoints</h3>
        <p><code>GET  /health</code> — API status</p>
        <p><code>POST /predict</code> — Predict customer segment</p>
        <p><code>GET  /segments</code> — Segment descriptions</p>
        <p><code>GET  /stats</code> — Model info</p>
        <p><a href="/docs">Open API Docs →</a></p>
    </div>
    <div class="box">
        <h3>Customer Segments</h3>
        <p>🔵 <b>High Value</b> — Premium customers, high spend</p>
        <p>🟢 <b>Potential</b> — Growing customers, moderate spend</p>
        <p>🟡 <b>At Risk</b> — Declining engagement, need win-back</p>
        <p>🔴 <b>Low Engagement</b> — Low spend, rarely responds</p>
    </div>
    <div class="box">
        <h3>Author</h3>
        <p>Habib Bashir Lawal — Junior Data Scientist</p>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=True)