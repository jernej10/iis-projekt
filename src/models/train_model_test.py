import os
from sklearn.model_selection import train_test_split
import onnxmltools
import mlflow
from mlflow import MlflowClient
from mlflow.onnx import log_model as log_onnx_model
from xgboost import XGBClassifier, XGBRegressor
from skl2onnx.common.data_types import FloatTensorType

from src.models.helpers.helper_dataset import load_dataset
import dagshub
import dagshub.auth as dh_auth
from dotenv import load_dotenv

from src.models.helpers.model_registry import empty_model_registry

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

load_dotenv()

def initialize_dagshub():
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    if mlflow.active_run():
        mlflow.end_run()

def load_and_prepare_data(file_path: str):
    dataset = load_dataset(file_path)
    dataset.sort_values(by="Date", inplace=True)
    dataset.drop(columns=["Date"], inplace=True)
    dataset.fillna(dataset.mean(), inplace=True)
    # drop last row

    dataset = dataset[:-1]
    return dataset

def split_data(dataset, predictors, target):
    X = dataset[predictors]
    y = dataset[target]
    return train_test_split(X, y, test_size=0.1, random_state=1)

def rename_columns(X):
    X.columns = [f'f{i}' for i in range(X.shape[1])]
    return X

def train_classification_model(X_train, y_train):
    model = XGBClassifier(random_state=1)
    model.fit(X_train, y_train)
    return model

def train_regression_model(X_train, y_train):
    model = XGBRegressor(random_state=1)
    model.fit(X_train, y_train)
    return model

def convert_to_onnx(model, input_shape):
    return onnxmltools.convert_xgboost(model, initial_types=[('input', FloatTensorType([None, input_shape]))])

def save_onnx_model(onnx_model, path):
    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())

def log_and_register_model(onnx_model, artifact_path, registered_model_name):
    mlflow.start_run(run_name=f"{registered_model_name}", nested=True)
    model_info = log_onnx_model(onnx_model, artifact_path=artifact_path, registered_model_name=registered_model_name)
    mlflow.end_run()
    return model_info

def transition_model_to_staging(client, model_name, model_info):
    mv = client.create_model_version(name=model_name, source=model_info.model_uri, run_id=model_info.run_id)
    client.transition_model_version_stage(model_name, mv.version, "staging")

def prepare_and_train_model(file: str):
    client = MlflowClient()

    dataset = load_and_prepare_data(file)
    predictors_cls = ["Close", "Volume", "Open", "High", "Low", "Open_Nasdaq"]
    predictors_reg = ["Close", "Volume", "Open", "High", "Low", "Open_Nasdaq"]

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_data(dataset, predictors_cls, "Target")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(dataset, predictors_reg, "Tomorrow")

    X_train_cls, X_test_cls = rename_columns(X_train_cls), rename_columns(X_test_cls)
    X_train_reg, X_test_reg = rename_columns(X_train_reg), rename_columns(X_test_reg)

    model_cls = train_classification_model(X_train_cls, y_train_cls)
    model_reg = train_regression_model(X_train_reg, y_train_reg)

    onnx_model_cls = convert_to_onnx(model_cls, len(predictors_cls))
    onnx_model_reg = convert_to_onnx(model_reg, len(predictors_reg))

    model_dir = "models/sp500"
    os.makedirs(model_dir, exist_ok=True)
    save_onnx_model(onnx_model_cls, os.path.join(model_dir, "model.onnx"))
    save_onnx_model(onnx_model_reg, os.path.join(model_dir, "model_regression.onnx"))

    model_cls_info = log_and_register_model(onnx_model_cls, "models/sp500", "sp500_model")
    model_reg_info = log_and_register_model(onnx_model_reg, "models/sp500/regression", "sp500_model_regression")

    transition_model_to_staging(client, "sp500_model", model_cls_info)
    transition_model_to_staging(client, "sp500_model_regression", model_reg_info)

    # Log the ONNX model params to MLflow
    mlflow.start_run(run_name=f"sp500", nested=True)
    mlflow.log_param("Params", "/")

    print(f"Models have been trained and saved!")

def main():
    file = "data/current_data.csv"

    initialize_dagshub()
    prepare_and_train_model(file)

    #empty_model_registry()

if __name__ == "__main__":
    main()
