import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import yfinance as yf
from src.models.helpers.model_registry import download_model, ModelType

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    sp500 = yf.Ticker(ticker)
    sp500 = sp500.history(period=period, interval=interval)
    return sp500


@app.get("/predict")
async def predict():
    df = fetch_stock_data("^GSPC", "1d", "1d")
    if df.empty:
        return {"error": "No data fetched from Yahoo Finance"}

    predictors = ["Close", "Volume", "Open", "High", "Low"]

    if not all(col in df.columns for col in predictors):
        return {"error": "Fetched data does not contain the required columns"}

    df = df.assign(Target=[0])

    X_test = df[predictors].values.astype(np.float32)

    production_model_path = download_model("sp500_model", ModelType.PRODUCTION)

    # Load the trained model
    model = ort.InferenceSession(production_model_path)

    # Check the model input name
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name

    try:
        predictions = model.run([output_name], {input_name: X_test})[0]
    except Exception as e:
        return {"error": str(e)}

    return {"prediction": predictions.tolist()}


@app.get("/")
def root():
    return {"message": "Hello, FastAPI!"}

# RUN in root -> uvicorn src.serve.main:app --reload --port 8000
