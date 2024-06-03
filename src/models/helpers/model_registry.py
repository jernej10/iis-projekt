import os
import dagshub.auth as dh_auth
import joblib
import onnx
import onnxruntime as ort
from enum import Enum, auto
from mlflow.onnx import load_model as load_onnx
from mlflow.sklearn import load_model as load_scaler
from dagshub.data_engine.datasources import mlflow
from mlflow import MlflowClient
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
import dagshub


load_dotenv()

def get_latest_model_version(model_name: str):
    try:
        client = MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=["staging"])[0]
        model_url = model_version.source
        model = load_onnx(model_url)
        return model
    except IndexError:
        print(f"Model {model_name} not found.")
        return None

def get_latest_scaler_version(model_name: str):
    try:
        client = MlflowClient()
        model_version = client.get_latest_versions(f"{model_name}_scaler", stages=["staging"])[0]
        model_url = model_version.source
        scaler = load_scaler(model_url)
        return scaler
    except IndexError:
        print(f"Scaler for model {model_name} not found.")
        return None

def get_production_model(model_name: str):
    try:
        client = MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=["production"])[0]
        model_url = model_version.source
        production_model = load_onnx(model_url)
        return production_model
    except IndexError:
        print(f"Production model {model_name} not found.")
        return None

def get_production_scaler(model_name: str):
    try:
        client = MlflowClient()
        model_version = client.get_latest_versions(f"{model_name}_scaler", stages=["production"])[0]
        model_url = model_version.source
        production_scaler = load_scaler(model_url)
        return production_scaler
    except IndexError:
        print(f"Production scaler for model {model_name} not found.")
        return None

class ModelType(Enum):
    LATEST = auto()
    PRODUCTION = auto()


def download_model(model_name: str, model_type: ModelType) -> str | None:
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    folder_name = f"models/sp500"
    model_type_str = model_type.name.lower()

    model_func = get_latest_model_version if model_type == ModelType.LATEST else get_production_model

    model = model_func(model_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if model is None:
        return None

    onnx.save_model(model, f"{folder_name}/{model_name}_{model_type.name.lower()}.onnx")
    print(f"{model_type_str.capitalize()} model for {model_name} has been downloaded.")
    model_path = f"{folder_name}/{model_name}_{model_type}.onnx"

    return model_path


def empty_model_registry():
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    client = MlflowClient()

    model_name_cls = f"sp500_model"
    client.delete_registered_model(model_name_cls)

    model_name_reg = f"sp500_model_regression"
    client.delete_registered_model(model_name_reg)
