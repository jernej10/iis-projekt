import os
import pandas as pd
from pymongo import MongoClient
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from dotenv import load_dotenv
import dagshub
from dagshub.data_engine.datasources import mlflow
import dagshub.auth as dh_auth

from src.data.helpers.send_email import send_email

load_dotenv()

# MongoDB connection setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client.get_database("db")
collection = db.get_collection("predictions")


def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    sp500 = yf.Ticker(ticker)
    sp500 = sp500.history(period=period, interval=interval)
    return sp500


def get_predictions():
    documents = collection.find()
    results = []
    for document in documents:
        document['_id'] = str(document['_id'])  # Convert ObjectId to string for JSON serialization
        results.append(document)
    return results


def main():
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    if mlflow.active_run():
        mlflow.end_run()

    # Fetch predictions from MongoDB
    predictions = get_predictions()
    print('Predictions:', predictions)

    # Fetch stock data
    data = fetch_stock_data("^GSPC", "5d", "1d")
    data['Tomorrow'] = data['Close'].shift(-1)
    data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)

    print('Stock actual data:')
    print(data.head())

    # Create a DataFrame from predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
    predictions_df['predictions'] = predictions_df['predictions'].apply(lambda x: x[0])
    predictions_df.drop(columns=['input_data', '_id'], inplace=True)

    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date
    predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp']).dt.date

    print('Predictions data:')
    print(predictions_df.head(20))

    # Merge stock data with predictions
    merged_df = pd.merge(data, predictions_df, left_on='Date', right_on='timestamp', how='inner')
    print('Merged data:')
    print(merged_df.head(20))

    # Calculate metrics
    y_true = merged_df['Target']
    y_pred = merged_df['predictions']

    # Start mlflow run
    mlflow.start_run(run_name="Production model evaluation")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Log metrics to mlflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    if accuracy < 0.5:
        print("Model is not performing well")
        send_email(f"Model in production is not performing well. Accuracy is: {accuracy:.2f}!")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # End mlflow run
    mlflow.end_run()


if __name__ == "__main__":
    main()
