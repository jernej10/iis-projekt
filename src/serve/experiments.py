import os
import mlflow
from dotenv import load_dotenv
import dagshub
from dagshub.data_engine.datasources import mlflow
import dagshub.auth as dh_auth

load_dotenv()



def get_metrics_history():
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    if mlflow.active_run():
        mlflow.end_run()

    metrics = {"classification": [], "regression": []}

    # Classification metrics
    runs = mlflow.search_runs(["1"])
    for index, row in runs.iterrows():
        metrics["classification"].append({
            "accuracy": row.get('metrics.accuracy', None),
            "precision": row.get('metrics.precision', None),
            "recall": row.get('metrics.recall', None)
        })

    # Regression metrics
    runs = mlflow.search_runs(["3"])
    for index, row in runs.iterrows():
        metrics["regression"].append({
            "mse": row.get('metrics.mse', None),
            "mae": row.get('metrics.mae', None),
            "evs": row.get('metrics.evs', None)
        })

    # End mlflow run
    mlflow.end_run()

    return metrics

def get_production_metrics_history():
    dh_auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init('iis-projekt', 'jernej10', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/jernej10/iis-projekt.mlflow')

    if mlflow.active_run():
        mlflow.end_run()

    metrics = {"classification": [], "regression": []}

    # Classification metrics
    runs = mlflow.search_runs(["4"])
    for index, row in runs.iterrows():
        metrics["classification"].append({
            "accuracy": row.get('metrics.accuracy', None),
            "precision": row.get('metrics.precision', None),
            "recall": row.get('metrics.recall', None),
            "f1": row.get('metrics.f1', None)
        })


    # End mlflow run
    mlflow.end_run()

    return metrics



if __name__ == "__main__":
    metrics = get_metrics_history()
    print(metrics)
    #print(get_production_metrics_history())