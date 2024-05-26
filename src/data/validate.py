import pandas as pd

def read_data(filename):
    return pd.read_csv(filename)

def validate(reference_data, current_data):
    # Check if headers match
    if not reference_data.columns.equals(current_data.columns):
        print("Imena stolpcev niso ista!")
        return False

    # Check if the number of columns match
    if reference_data.shape[1] != current_data.shape[1]:
        print("Število stolpcev se ne ujema!")
        return False

    # Check if data types match
    if not (reference_data.dtypes == current_data.dtypes).all():
        print("Tipi podatkov niso enaki!")
        return False

    # Check if "Target" column is [0,1]
    target_column = 'Target'
    if target_column in reference_data.columns:
        valid_target_values = {0, 1}
        if not set(reference_data[target_column]).issubset(valid_target_values):
            print("Referenčni podatki imajo neveljavne vrednosti v stolpcu 'Target'!")
            return False
        if not set(current_data[target_column]).issubset(valid_target_values):
            print("Trenutni podatki imajo neveljavne vrednosti v stolpcu 'Target'!")
            return False
    else:
        print(f"Manjka stolpec 'Target' v podatkih!")
        return False

    print("Validacija uspešna!")
    return True

def main():
    reference_data = read_data('data/reference_data.csv')
    current_data = read_data('data/current_data.csv')
    validate(reference_data, current_data)

if __name__ == "__main__":
    main()
