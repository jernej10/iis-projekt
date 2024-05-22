import pandas as pd
import os

def process_data(data: pd.DataFrame, directory: str, filename: str) -> None:
    data['Date'] = data['Date'].str.split(' ').str[0]
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data[data['Date'] > pd.to_datetime('1990-01-01')]

    if data is not None:
        raw_data_directory = directory
        if not os.path.exists(raw_data_directory):
            os.makedirs(raw_data_directory)
        filename = f"{filename}.csv"
        file_path = os.path.join(raw_data_directory, filename)
        data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
        data['Tomorrow'] = data['Close'].shift(-1)
        data['Target'] = (data['Tomorrow'] > data['Close']).astype(int)
        data.to_csv(file_path, index=True)

def main():
    sp500 = pd.read_csv("data/raw/stock/sp500.csv")
    process_data(sp500, "data/processed/stock", "sp500")


if __name__ == "__main__":
    main()