import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import yfinance as yf
from pydantic import BaseModel

from src.models.helpers.model_registry import download_model, ModelType
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from fastapi.staticfiles import StaticFiles

from src.serve.experiments import get_metrics_history, get_production_metrics_history

load_dotenv()
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client.get_database("db")
collection = db.get_collection("predictions")
validation_results_collection = db.get_collection("validation-results")
metric_limit_collection = db.get_collection("metric-limit")

def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.Ticker(ticker)
    data = data.history(period=period, interval=interval)
    data.reset_index(inplace=True)  # Resetiranje indeksa

    return data

# Mount the /img directory to serve static files
# TODO fix url for production
app.mount("/img", StaticFiles(directory="src/serve/img"), name="img")

class MetricLimit(BaseModel):
    value: float

@app.post("/metric-limit")
async def create_metric_limit(metric_limit: MetricLimit):
    try:
        metric_limit_collection.insert_one(metric_limit.dict())
        return {"message": "Metric limit added successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/metric-limit/latest")
async def get_latest_metric_limit():
    try:
        latest_metric_limit = metric_limit_collection.find().sort("_id", -1).limit(1)
        latest_metric_limit = list(latest_metric_limit)
        if latest_metric_limit:
            latest_metric_limit[0]["_id"] = str(latest_metric_limit[0]["_id"])
            return {"latest_metric_limit": latest_metric_limit[0]}
        else:
            return {"message": "No metric limit found"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict")
async def predict():
    df = fetch_stock_data("^GSPC", "1d", "1d")
    if df.empty:
        return {"error": "No data fetched from Yahoo Finance"}

    # Pridobitev podatkov Nasdaq 100
    nasdaq_data = fetch_stock_data("^NDX", "1d", "1d")

    df["Open_Nasdaq"] = nasdaq_data["Open"]

    predictors = ["Close", "Volume", "Open", "High", "Low", "Open_Nasdaq"]

    if not all(col in df.columns for col in predictors):
        return {"error": "Fetched data does not contain the required columns"}

    df = df.assign(Target=0)

    print(df.head())

    X_test = df[predictors]
    X_test.columns = [f'f{i}' for i in range(X_test.shape[1])]


    production_model_path = download_model("sp500_model", ModelType.PRODUCTION)
    model = ort.InferenceSession("../../models/sp500/sp500_model_production.onnx")

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[1].name

    try:
        predictions = model.run([output_name], {input_name: X_test.values.astype(np.float32)})[0]
        predicted_classes = [1 if prediction[1] > 0.6 else 0 for prediction in predictions]
    except Exception as e:
        return {"error": str(e)}

    prediction_result = predicted_classes

    # Save to MongoDB
    document = {
        "timestamp": datetime.now().isoformat(),
        "input_data": df[predictors].to_dict(orient="records"),
        "predictions": prediction_result
    }
    collection.insert_one(document)

    return {"prediction": prediction_result}

@app.get("/predict/regression")
async def predict_regression():
    df = fetch_stock_data("^GSPC", "1d", "1d")
    if df.empty:
        return {"error": "No data fetched from Yahoo Finance"}

    # Pridobitev podatkov Nasdaq 100
    nasdaq_data = fetch_stock_data("^NDX", "1d", "1d")

    df["Open_Nasdaq"] = nasdaq_data["Open"]

    predictors = ["Close", "Volume", "Open", "High", "Low", "Open_Nasdaq"]

    if not all(col in df.columns for col in predictors):
        return {"error": "Fetched data does not contain the required columns"}

    # Dodaj stolpec za ciljni atribut in nastavi vrednosti na 0
    df = df.assign(Tomorrow=0)

    print(df.head())

    # Pripravi podatke za napovedovanje
    X_test = df[predictors]
    X_test.columns = [f'f{i}' for i in range(X_test.shape[1])]

    X_test = df[predictors].values.astype(np.float32)

    production_model_path = download_model("sp500_model_regression", ModelType.PRODUCTION)

    # Load the trained model
    model = ort.InferenceSession("../../models/sp500/sp500_model_regression_production.onnx")

    # Check the model input name
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    try:
        predictions = model.run([output_name], {input_name: X_test})[0]
    except Exception as e:
        return {"error": str(e)}

    return {"prediction": predictions.tolist()}

@app.get("/historical-prices")
async def historical_prices():
    df = fetch_stock_data("^GSPC", "1y", "1d")  # Fetch 1 year of daily data
    if df.empty:
        return {"error": "No data fetched from Yahoo Finance"}

    # Ensure necessary columns are present
    if "Close" not in df.columns:
        return {"error": "Fetched data does not contain the required columns"}

    # Convert the DataFrame to a suitable format for the frontend
    prices = df.reset_index()[["Date", "Close"]]
    prices["Date"] = prices["Date"].dt.strftime('%Y-%m-%d')
    prices_list = prices.to_dict(orient="records")

    return {"prices": prices_list}

@app.get("/latest-validation-result")
async def get_latest_validation_result():
    result = validation_results_collection.find_one(sort=[("timestamp", -1)])
    if result:
        result["_id"] = str(result["_id"])
    return result

@app.get("/metrics-history")
async def metrics():
    return get_metrics_history()

@app.get("/production-metrics-history")
async def production_metrics():
    return get_production_metrics_history()

@app.get("/")
def root():
    return {"message": "Hello, FastAPI!"}

# RUN in root -> uvicorn src.serve.main:app --reload --port 8000
