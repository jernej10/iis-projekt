import os
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def read_data(filename):
    return pd.read_csv(filename)


def validate(reference_data, current_data):
    validation_result = {"success": True, "messages": []}

    # Check if headers match
    if not reference_data.columns.equals(current_data.columns):
        validation_result["success"] = False
        validation_result["messages"].append("Imena stolpcev niso enaka!")
    else:
        validation_result["messages"].append("Imena stolpcev so enaka!")

    # Check if the number of columns match
    if reference_data.shape[1] != current_data.shape[1]:
        validation_result["success"] = False
        validation_result["messages"].append("Število stolpcev se ne ujema!")
    else:
        validation_result["messages"].append("Število stolpcev se ujema!")

    # Check if data types match
    if not (reference_data.dtypes == current_data.dtypes).all():
        validation_result["success"] = False
        validation_result["messages"].append("Tipi podatkov niso enaki!")
    else:
        validation_result["messages"].append("Tipi podatkov so enaki!")

    # Check if "Target" column is [0,1]
    target_column = 'Target'
    if target_column in reference_data.columns:
        valid_target_values = {0, 1}

        if not set(current_data[target_column]).issubset(valid_target_values):
            validation_result["success"] = False
            validation_result["messages"].append("Trenutni podatki imajo neveljavne vrednosti v stolpcu 'Target'!")
        else:
            validation_result["messages"].append("Trenutni podatki imajo veljavne vrednosti v stolpcu 'Target'!")
    else:
        validation_result["success"] = False
        validation_result["messages"].append("Manjka stolpec 'Target' v podatkih!")

    return validation_result


def main():
    # MongoDB connection setup
    MONGO_URI = os.getenv("MONGO_URI")
    client = MongoClient(MONGO_URI)
    db = client.get_database("db")
    collection = db.get_collection("validation-results")

    reference_data = read_data('data/reference_data.csv')
    current_data = read_data('data/current_data.csv')
    result = validate(reference_data, current_data)

    # Add timestamp to the result
    result["timestamp"] = datetime.now().isoformat()

    collection.insert_one(result)


if __name__ == "__main__":
    main()