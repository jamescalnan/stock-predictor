
# Stock Predictor

## Overview

The Stock Predictor project aims to predict stock prices using machine learning algorithms and sentiment analysis. It incorporates historical stock data along with news sentiment to make more accurate predictions. The project primarily uses Gradient Boosting Regressor for prediction and VADER for sentiment analysis.

## Features

- Predicts the next day's stock closing price
- Uses historical stock data including Open, High, Low, Close, Volume, etc.
- Incorporates sentiment analysis on news articles related to the stock
- Uses Gradient Boosting Regressor for machine learning
- Includes a rich logging and console output for easy debugging and interpretation

## File Structure

\```
stock-predictor/
|-- .vscode/
|   |-- settings.json
|-- historical_data/
|   |-- (Various CSV files for different stocks)
|-- news_sentiment_data/
|   |-- (CSV files containing news sentiment scores)
|-- unused_strategies/
|   |-- (Various Python scripts for unused strategies)
|-- LICENSE
|-- main.py
|-- ml_feature_log.py   <- Main running script
|-- news_sentiment.py
|-- README.md
\```

## Setup and Installation

1. Clone the repository: `git clone https://github.com/jamescalnan/stock-predictor.git`
2. Navigate to the project folder: `cd stock-predictor`
3. Install the required Python packages: `pip install -r requirements.txt`

## How to Run

1. Navigate to the project folder in the terminal.
2. Run the script: `python ml_feature_log.py`
3. Follow the on-screen instructions to proceed.


This is the main script where data is processed and predictions are made. Key functions include:

- `compute_rsi(data)`: Computes the Relative Strength Index.
- `compute_macd(data)`: Computes the Moving Average Convergence Divergence.
- `compute_bollinger_bands(data)`: Computes the Bollinger Bands.
- `compute_stochastic_oscillator(data)`: Computes the Stochastic Oscillator.
- `GBC_Train(data, ticker)`: Trains the Gradient Boosting Regressor model and returns it along with other metrics.
- `predict_next_day_closing(data, model, given_date)`: Predicts the closing price for the next day after a given date.
- `compare_predictions_with_actual(data, model)`: Compares predicted and actual closing prices for all dates in the dataset.

## Logging

The project uses the `rich` library for enhanced logging.

- Financial data is for educational purposes only and should not be considered as financial advice.

## Author

James Calnan

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
