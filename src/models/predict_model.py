import os
import numpy as np
import mlflow
from mlflow import MlflowClient
import onnxruntime as ort

from src.models.helpers.helper_dataset import load_dataset, write_metrics_to_file, write_regression_metrics_to_file
from src.models.helpers.helper_training import evaluate_model_performance_classification, evaluate_model_performance_regression
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
    X_test = test_dataset.drop(columns=["Target"])

    production_model_path = download_model(model_name, ModelType.PRODUCTION)
    latest_model_path = download_model(model_name, ModelType.LATEST)

    production_model_path_regression = download_model(model_name + "_regression", ModelType.PRODUCTION)
    latest_model_path_regression = download_model(model_name + "_regression", ModelType.LATEST)

    if latest_model_path is None:
        print("No latest model found.")
        return

    if production_model_path is None:
        update_production_model(model_name)
        return

    if latest_model_path_regression is None:
        print("No latest regression model found.")
        return

    if production_model_path_regression is None:
        update_production_model(model_name + "_regression")
        return

    latest_model = ort.InferenceSession(latest_model_path)
    production_model = ort.InferenceSession(production_model_path)

    latest_model_regression = ort.InferenceSession(latest_model_path_regression)
    production_model_regression = ort.InferenceSession(production_model_path_regression)

    print("Classification model")
    for node in latest_model.get_outputs():
        print(node.name)

    print("Regression model")
    for node in latest_model_regression.get_outputs():
        print(node.name)

    input_name = latest_model.get_inputs()[0].name
    label_name = latest_model.get_outputs()[0].name
    label_name_probability = latest_model.get_outputs()[1].name

    input_name_regression = latest_model_regression.get_inputs()[0].name
    label_name_regression = latest_model_regression.get_outputs()[0].name

    predictors = ["Close", "Volume", "Open", "High", "Low"]
    X_test_classification = X_test[predictors].values
    X_test_classification = X_test_classification.astype(np.float32)
    latest_model_predictions = latest_model.run([label_name_probability], {input_name: X_test_classification})[0]

    predictors_regression = ["Close", "Volume", "High", "Low"]
    X_test_regression = X_test[predictors_regression].values
    X_test_regression = X_test_regression.astype(np.float32)
    latest_model_predictions_regression = latest_model_regression.run([label_name_regression], {input_name_regression: X_test_regression})[0]
    print('latest_model_predictions_regression', latest_model_predictions_regression)
    # Determine predicted class based on the probability with a threshold
    predicted_classes = [1 if prediction[1] > 0.35 else 0 for prediction in latest_model_predictions]
    print(predicted_classes)

    accuracy_test, precision_test, recall_test, f1_test = evaluate_model_performance_classification(test_dataset["Target"], predicted_classes)
    mse_test, mae_test, evs_test = evaluate_model_performance_regression(test_dataset["Open"], latest_model_predictions_regression)

    print(f"Test metrics for {model_name} have been calculated!")
    print(f"Accuracy: {accuracy_test}")
    print(f"Precision: {precision_test}")
    print(f"Recall: {recall_test}")
    print(f"F1: {f1_test}")

    print(f"Test metrics for {model_name}_regression have been calculated!")
    print(f"MSE: {mse_test}")
    print(f"MAE: {mae_test}")
    print(f"EVS: {evs_test}")

    mlflow.start_run(run_name=f"Prediction for {model_name}", nested=True)
    mlflow.log_metric("accuracy", accuracy_test)
    mlflow.log_metric("precision", precision_test)
    mlflow.log_metric("recall", recall_test)
    mlflow.log_metric("f1", f1_test)

    mlflow.start_run(run_name=f"Prediction for {model_name}_regression", nested=True)
    mlflow.log_metric("mse", mse_test)
    mlflow.log_metric("mae", mae_test)
    mlflow.log_metric("evs", evs_test)


    production_model_predictions = production_model.run([label_name_probability], {input_name: X_test_classification})[0]
    predicted_classes = [1 if prediction[1] > 0.35 else 0 for prediction in production_model_predictions]

    accuracy_production, precision_production, recall_production, f1_production = evaluate_model_performance_classification(test_dataset["Target"], predicted_classes)

    production_model_predictions_regression = production_model_regression.run([label_name_regression], {input_name_regression: X_test_regression})[0]
    mse_production, mae_production, evs_production = evaluate_model_performance_regression(test_dataset["Open"], production_model_predictions_regression)

    # Set model to production if it performs better
    if precision_test > precision_production:
        update_production_model(model_name)

    if mse_test < mse_production:
        update_production_model(model_name + "_regression")

    write_metrics_to_file(f"reports/{model_name}/metrics.txt", "RandomForestClassifier", accuracy_test, precision_test, recall_test, f1_test)
    write_regression_metrics_to_file(f"reports/{model_name}_regression/metrics.txt", "RandomForestRegressor", mse_test, mae_test, evs_test)

    print(f"Test metrics for {model_name} have been calculated!")
    print(f"Test metrics for {model_name}_regression have been calculated!")

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
