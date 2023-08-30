import requests
import datetime

from rich.console import Console

console = Console()

def fetch_stock_data(symbol, target_date):
    API_URL = "https://www.alphavantage.co/query"
    API_KEY = "XJSD1DOD6XJK3S08"

    parameters = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "date": target_date,
        "interval": "60min",
        "apikey": API_KEY
    }

    response = requests.get(API_URL, params=parameters)
    data = response.json()

    console.print(data)


if __name__ == '__main__':
    target_date = "2022-08-15"  # Replace with the date you're interested in, in YYYY-MM-DD format
    fetch_stock_data('AAPL', target_date)
