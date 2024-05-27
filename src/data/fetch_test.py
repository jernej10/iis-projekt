import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    sp500 = yf.Ticker(ticker)
    sp500 = sp500.history(period=period, interval=interval)
    return sp500

def main():
    df = fetch_stock_data("^GSPC", "1d", "1d")
    print(df.head())


if __name__ == "__main__":
    main()

# ^GSPC: Period is invalid, must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
# ^GSPC: Invalid input - interval is not supported. Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]