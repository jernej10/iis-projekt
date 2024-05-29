import os

import pandas as pd


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv("data/processed/mbajk_processed.csv")
    dataset.drop(columns=["date"], inplace=True)
    return dataset


def load_dataset(file: str) -> pd.DataFrame:
    dataset = pd.read_csv(file)
    return dataset


def write_metrics_to_file(file_path: str, model_name: str, accuracy: float, precision: float, recall: float, f1: float) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1 Score: {f1}\n")

def write_regression_metrics_to_file(file_path: str, model_name: str, mse: float, mae: float, evs: float) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        file.write(f"Model: {model_name}\n")
        file.write(f"MSE: {mse}\n")
        file.write(f"MAE: {mae}\n")
        file.write(f"EVS: {evs}\n")

