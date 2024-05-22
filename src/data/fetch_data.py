import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    sp500 = yf.Ticker(ticker)
    sp500 = sp500.history(period=period, interval=interval)
    return sp500

def save_data(data: pd.DataFrame, directory: str, filename: str) -> None:
    if data is not None:
        raw_data_directory = directory
        if not os.path.exists(raw_data_directory):
            os.makedirs(raw_data_directory)
        filename = f"{filename}.csv"
        file_path = os.path.join(raw_data_directory, filename)
        file_exists = os.path.isfile(file_path)
        print(data.head())
        if file_exists:
            data.to_csv(file_path, mode='a', header=False, index=True)
        else:
            data.to_csv(file_path, index=True)

def main():
    df = fetch_stock_data("^GSPC", "1d", "1d")
    save_data(df, "../../data/raw/stock", "sp500")


if __name__ == "__main__":
    main()

# ^GSPC: Period is invalid, must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
# ^GSPC: Invalid input - interval is not supported. Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]