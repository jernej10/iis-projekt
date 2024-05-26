import os
from glob import glob
import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient
import onnxruntime as ort
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score

from src.models.helpers.helper_dataset import load_dataset, write_metrics_to_file
from src.models.helpers.helper_training import prepare_validation_model_data, evaluate_model_performance_classification
from src.models.helpers.model_registry import download_model, ModelType

from dotenv import load_dotenv
import dagshub
from dagshub.data_engine.datasources import mlflow
import dagshub.auth as dh_auth

load_dotenv()

def update_production_model(model_name: str) -> None:
    client = MlflowClient()
    new_model_version = client.get_latest_versions(model_name, stages=["staging"])[0].version
    client.transition_model_version_stage(model_name, new_model_version, "production")
    print(f"[Update Model] - New model for {model_name} has been set to production")

def predict_model(model_name: str, test_file: str):
    test_dataset = load_dataset(test_file)
    test_dataset.sort_values(by="Date", inplace=True)
    test_dataset.drop(columns=["Date"], inplace=True)
    X_test = test_dataset.drop(columns=["Target"]).values
    y_test = test_dataset["Target"].values

    production_model_path = download_model(model_name, ModelType.PRODUCTION)
    latest_model_path = download_model(model_name, ModelType.LATEST)

    if latest_model_path is None:
        print("No latest model found.")
        return

    if production_model_path is None:
        update_production_model(model_name)
        return

    latest_model = ort.InferenceSession(latest_model_path)
    production_model = ort.InferenceSession(production_model_path)

    # test
    for node in latest_model.get_outputs():
        print(node.name)

    predictors = ["Close", "Volume", "Open", "High", "Low"]
    X_test = X_test.astype(np.float32)
    # Get the column indices corresponding to the predictor columns
    column_indices = [test_dataset.columns.get_loc(col) for col in predictors]

    # Select the predictor columns from X_test using column indices
    X_test_predictors = X_test[:, column_indices]
    latest_model_predictions = latest_model.run(["output_label"], {"input": X_test_predictors})[0]

    accuracy_test, precision_test, recall_test, f1_test = evaluate_model_performance_classification(y_test, latest_model_predictions)

    mlflow.start_run(run_name=f"Prediction for {model_name}", nested=True)
    mlflow.log_metric("accuracy", accuracy_test)
    mlflow.log_metric("precision", precision_test)
    mlflow.log_metric("recall", recall_test)
    mlflow.log_metric("f1", f1_test)


    production_model_predictions = production_model.run(["output_label"], {"input": X_test_predictors})[0]
    accuracy_production, precision_production, recall_production, f1_production = evaluate_model_performance_classification(y_test, production_model_predictions)

    # Set model to production if it performs better
    if precision_test > precision_production:
        update_production_model(model_name)

    write_metrics_to_file(f"reports/{model_name}/metrics.txt", "RandomForest", accuracy_test, precision_test, recall_test, f1_test)

    print(f"Test metrics for {model_name} have been calculated!")

    mlflow.end_run()

def main():
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    if mlflow.active_run():
        mlflow.end_run()

    model_name = "sp500_model"
    predict_model(model_name, "data/validation/test.csv")

if __name__ == "__main__":
    main()
