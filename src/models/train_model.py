import os
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import mlflow
from mlflow.onnx import log_model as log_onnx_model
from src.models.helpers.helper_dataset import load_dataset
import dagshub
from dagshub.data_engine.datasources import mlflow
import dagshub.auth as dh_auth
from dotenv import load_dotenv

load_dotenv()

def prepare_and_train_model() -> None:
    client = MlflowClient()

    dataset = load_dataset(f"data/validation/train.csv")
    dataset.sort_values(by="Date", inplace=True)
    dataset.drop(columns=["Date"], inplace=True)

    # fill missing values with average values
    dataset.fillna(dataset.mean(), inplace=True)

    n_estimators = 200
    min_samples_split = 50
    random_state = 1

    predictors = ["Close", "Volume", "Open", "High", "Low"]
    predictors_regression = ["Close", "Volume", "High", "Low"]

    # Classification model
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
    model.fit(dataset[predictors], dataset["Target"])

    # Regression model
    model_regression = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
    model_regression.fit(dataset[predictors_regression], dataset["Open"])

    # Convert RandomForest model to ONNX
    initial_type = [('input', FloatTensorType([None, len(predictors)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    initial_type_regression = [('input', FloatTensorType([None, len(predictors_regression)]))]
    onnx_model_regression = convert_sklearn(model_regression, initial_types=initial_type_regression)

    # Create the directory if it doesn't exist
    model_dir = "models/sp500"
    os.makedirs(model_dir, exist_ok=True)
    model_onnx_path = os.path.join(model_dir, "model.onnx")
    model_onnx_path_regression = os.path.join(model_dir, "model_regression.onnx")

    # Save the ONNX model to a file
    with open(model_onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    with open(model_onnx_path_regression, "wb") as f:
        f.write(onnx_model_regression.SerializeToString())

    # Log the ONNX model to MLflow
    mlflow.start_run(run_name=f"sp500", nested=True)

    model_ = log_onnx_model(onnx_model, artifact_path="models/sp500", registered_model_name="sp500_model")
    model_regression_ = log_onnx_model(onnx_model_regression, artifact_path="models/sp500/regression", registered_model_name="sp500_model_regression")

    mv = client.create_model_version(name="sp500_model", source=model_.model_uri, run_id=model_.run_id)
    client.transition_model_version_stage("sp500_model", mv.version, "staging")

    mv_regression = client.create_model_version(name="sp500_model_regression", source=model_regression_.model_uri, run_id=model_regression_.run_id)
    client.transition_model_version_stage("sp500_model_regression", mv_regression.version, "staging")

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("min_samples_split", min_samples_split)
    mlflow.log_param("random_state", random_state)

    print(f"Models has been trained and saved!")

    mlflow.end_run()

def main():
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    if mlflow.active_run():
        mlflow.end_run()

    prepare_and_train_model()

if __name__ == "__main__":
    main()
