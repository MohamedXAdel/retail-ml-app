from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Customer Segmentation ML App")

# Mount static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Cluster names
CLUSTER_NAMES = {
    0: "Loyal Regulars",
    1: "At-Risk Customers",
    2: "High-Value Loyal",
    3: "VIP Whales"
}

# Pydantic model for input validation
class RFMInput(BaseModel):
    recency: float
    frequency: float
    monetary: float

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(request: Request, recency: float = Form(...), frequency: float = Form(...), monetary: float = Form(...)):
    try:
        # Prepare input
        features = np.array([[recency, frequency, monetary]])
        features_scaled = scaler.transform(features)
        
        # Predict cluster
        cluster = int(kmeans.predict(features_scaled)[0])
        segment = CLUSTER_NAMES.get(cluster, "Unknown")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": True,
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary,
            "cluster": cluster,
            "segment": segment
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Invalid input. Please enter positive numbers."
        })