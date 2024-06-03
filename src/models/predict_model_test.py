import os
import numpy as np
import mlflow
import onnxruntime as ort

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from mlflow import MlflowClient

from src.models.helpers.helper_dataset import load_dataset, write_metrics_to_file, write_regression_metrics_to_file
from src.models.helpers.helper_training import evaluate_model_performance_classification, \
    evaluate_model_performance_regression
from src.models.helpers.model_registry import download_model, ModelType

import dagshub
import dagshub.auth as dh_auth

load_dotenv()


def update_production_model(model_name: str) -> None:
    client = MlflowClient()
    new_model_version = client.get_latest_versions(model_name, stages=["staging"])[0].version
    client.transition_model_version_stage(model_name, new_model_version, "production")
    print(f"[Update Model] - New model for {model_name} has been set to production")


def load_and_prepare_data(test_file: str, predictors: list, target: str):
    test_dataset = load_dataset(test_file)
    test_dataset.dropna(inplace=True)

    X = test_dataset[predictors]
    y = test_dataset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

    X_test.columns = [f'f{i}' for i in range(X_test.shape[1])]

    return X_test, y_test, test_dataset


def evaluate_classification(model_name: str, test_file: str):
    #predictors_classification = ["Close", "Volume", "Open", "High", "Low"]
    predictors_classification = ["Close", "Volume", "Open", "High", "Low", "Open_Nasdaq"]


    X_test, y_test, test_dataset = load_and_prepare_data(test_file, predictors_classification, target="Target")

    production_model_path = download_model(model_name, ModelType.PRODUCTION)
    latest_model_path = download_model(model_name, ModelType.LATEST)

    if production_model_path is None:
        update_production_model(model_name)
        return

    latest_model = ort.InferenceSession(f"models/sp500/{model_name}_latest.onnx")
    input_name = latest_model.get_inputs()[0].name
    label_name_probability = latest_model.get_outputs()[1].name

    latest_model_predictions = latest_model.run([label_name_probability], {input_name: X_test.values.astype(np.float32)})[0]
    predicted_classes = [1 if prediction[1] > 0.6 else 0 for prediction in latest_model_predictions]
    print(predicted_classes)
    accuracy, precision, recall, f1 = evaluate_model_performance_classification(y_test.values[-len(predicted_classes):],
                                                                                predicted_classes)

    print(f"Classification metrics for {model_name}:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Log metrics to MLflow
    mlflow.start_run(run_name=f"Prediction for {model_name}", nested=True)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # Get production model performance
    production_model = ort.InferenceSession("models/sp500/sp500_model_production.onnx")
    production_model_predictions = production_model.run([production_model.get_outputs()[1].name], {input_name: X_test.values.astype(np.float32)})[0]
    production_predicted_classes = [1 if prediction[1] > 0.6 else 0 for prediction in production_model_predictions]

    _, precision_production, _, _ = evaluate_model_performance_classification(y_test.values, production_predicted_classes)

    # Update production model if the new model performs better
    if precision > precision_production:
        update_production_model(model_name)

    write_metrics_to_file(f"reports/{model_name}/metrics.txt", "XGBClassifier", accuracy, precision, recall, f1)


def evaluate_regression(model_name: str, test_file: str):
    #predictors_regression = ["Close", "Volume", "High", "Low"]
    predictors_regression = ["Close", "Volume", "Open", "High", "Low", "Open_Nasdaq"]

    X_test, y_test, test_dataset = load_and_prepare_data(test_file, predictors_regression, target="Tomorrow")

    production_model_path = download_model(model_name, ModelType.PRODUCTION)
    latest_model_path = download_model(model_name, ModelType.LATEST)

    if production_model_path is None:
        update_production_model(model_name)
        return

    latest_model = ort.InferenceSession(f"models/sp500/{model_name}_latest.onnx")
    input_name = latest_model.get_inputs()[0].name
    label_name_regression = latest_model.get_outputs()[0].name

    latest_model_predictions = \
    latest_model.run([label_name_regression], {input_name: X_test.values.astype(np.float32)})[0]
    print(latest_model_predictions)
    mse, mae, evs = evaluate_model_performance_regression(y_test.values[-len(latest_model_predictions):],
                                                          latest_model_predictions)

    print(f"Regression metrics for {model_name}:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"EVS: {evs}")

    # Log metrics to MLflow
    mlflow.start_run(run_name=f"Prediction for {model_name}", nested=True)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("evs", evs)

    # Get production model performance
    production_model = ort.InferenceSession("models/sp500/sp500_model_regression_production.onnx")
    production_model_predictions = production_model.run([label_name_regression], {input_name: X_test.values.astype(np.float32)})[0]
    mse_production, _, _ = evaluate_model_performance_regression(y_test.values, production_model_predictions)

    # Update production model if the new model performs better
    if mse < mse_production:
        update_production_model(model_name)

    write_regression_metrics_to_file(f"reports/{model_name}/metrics.txt", "XGBRegressor", mse, mae, evs)


def main():
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    if mlflow.active_run():
        mlflow.end_run()

    file = "data/current_data.csv"

    evaluate_classification("sp500_model", file)
    evaluate_regression("sp500_model_regression", file)

    print(f"Test metrics for models have been calculated!")

    mlflow.end_run()


if __name__ == "__main__":
    main()
