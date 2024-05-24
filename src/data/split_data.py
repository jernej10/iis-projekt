import os
import pandas as pd

def split_data():
    current_data = pd.read_csv('data/current_data.csv')

    test_size = int(0.1 * len(current_data))

    test_data = current_data.tail(test_size)
    train_data = current_data.iloc[:-test_size]

    test_data.to_csv('data/validation/test.csv', index=False)
    train_data.to_csv('data/validation/train.csv', index=False)

def main():
    validation_directory = 'data/validation'
    if not os.path.exists(validation_directory):
        os.makedirs(validation_directory)

    split_data()

if __name__ == "__main__":
    main()
