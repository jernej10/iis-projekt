import pandas as pd
import os
import yfinance as yf

def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data

def process_data(sp500_data: pd.DataFrame, nasdaq_data: pd.DataFrame, directory: str, filename: str) -> None:
    sp500_data.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
    sp500_data['Tomorrow'] = sp500_data['Close'].shift(-1)
    sp500_data['Target'] = (sp500_data['Tomorrow'] > sp500_data['Close']).astype(int)

    horizons = [2, 5, 60, 250, 1000]

    for horizon in horizons:
        rolling_averages = sp500_data.drop(columns=['Date']).rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        sp500_data[ratio_column] = sp500_data["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        sp500_data[trend_column] = sp500_data.drop(columns=['Date']).shift(1).rolling(horizon).sum()["Target"]

    sp500_data['Date'] = sp500_data['Date'].str.split(' ').str[0]
    sp500_data['Date'] = pd.to_datetime(sp500_data['Date'], format='%Y-%m-%d')
    sp500_data = sp500_data[sp500_data['Date'] > pd.to_datetime('1990-01-01')]

    # Pridobitev Nasdaqa iz yfinance
    nasdaq_data = nasdaq_data.reset_index()
    nasdaq_data['Date'] = pd.to_datetime(nasdaq_data['Date']).dt.tz_localize(None)

    # Pretvorba S&P 500 datuma v enako časovno cono
    sp500_data['Date'] = sp500_data['Date'].dt.tz_localize(None)

    # Združite podatke S&P 500 in Nasdaq 100 na osnovi datuma
    sp500_data = pd.merge(sp500_data, nasdaq_data[['Date', 'Open']], on='Date', how='left', suffixes=('', '_Nasdaq'))
    sp500_data.rename(columns={'Open_x': 'Open', 'Open_y': 'Open_nasdaq'}, inplace=True)

    if sp500_data is not None:
        raw_data_directory = directory
        if not os.path.exists(raw_data_directory):
            os.makedirs(raw_data_directory)
        filename = f"{filename}.csv"
        file_path = os.path.join(raw_data_directory, filename)

        sp500_data.to_csv(file_path, index=True)

def main():
    sp500 = pd.read_csv("data/raw/stock/sp500.csv")
    nasdaq100 = fetch_stock_data("^NDX", "max", "1d")  # Nasdaq 100 index

    process_data(sp500, nasdaq100, "data/processed/stock", "sp500")

if __name__ == "__main__":
    main()
