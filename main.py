import pandas as pd
import numpy as np
from rich.console import Console
c = Console()
c.clear()

# Load the data
data = pd.read_csv('GOOGL.csv')
data['Return'] = data['Close'].pct_change()
data = data.dropna()

# Calculate drift and volatility
drift = data['Return'].mean()
volatility = data['Return'].std()

def gbm_simulation(S0, drift, volatility, days=1, simulations=10000):
    """Simulates stock prices for a specified number of days using Geometric Brownian Motion."""
    dt = 1  # time increment (1 day)
    simulated_prices = np.zeros((days, simulations))
    simulated_prices[0] = S0
    for day in range(1, days):
        dS = simulated_prices[day-1] * (drift * dt + volatility * np.random.normal(0, 1, simulations) * np.sqrt(dt))
        simulated_prices[day] = simulated_prices[day-1] + dS
    return simulated_prices

def backtest_gbm(data, drift, volatility, forecast_days=5):
    """Backtests the GBM prediction method over the entire dataset."""
    correct_predictions = 0
    total_predictions = len(data) - forecast_days
    for i in range(total_predictions):
        simulated_prices = gbm_simulation(data['Close'].iloc[i], drift, volatility, days=forecast_days)
        predicted_direction = "Increase" if np.mean(simulated_prices[-1]) > data['Close'].iloc[i] else "Decrease"
        actual_direction = "Increase" if data['Close'].iloc[i + forecast_days] > data['Close'].iloc[i] else "Decrease"
        c.print(f"Predicted: {predicted_direction} | Actual: {actual_direction}")
        
        if predicted_direction == actual_direction:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy

# Backtest the GBM prediction method for 5 days ahead
accuracy_gbm = backtest_gbm(data, drift, volatility, forecast_days=5)
c.print(f"Accuracy: {accuracy_gbm:.2%}")
