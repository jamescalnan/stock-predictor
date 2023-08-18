import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from rich.progress import track

# Load the dataset
data_googl = pd.read_csv('GOOGL.csv')

def backtest_arima_updated_v2(data, order=(1, 1, 1), start_point=100):
    """
    Backtests the ARIMA prediction method over the entire dataset.
    """
    correct_predictions = 0
    total_predictions = len(data) - start_point - 1
    predictions = []
    
    for i in track(range(start_point, len(data) - 1)):
        model = ARIMA(data[:i+1], order=order)
        model_fit = model.fit()
        prediction = model_fit.forecast(steps=1)[0]
        predicted_direction = "Increase" if prediction > data[i] else "Decrease"
        actual_direction = "Increase" if data[i + 1] > data[i] else "Decrease"
        
        predictions.append(predicted_direction)
        
        if predicted_direction == actual_direction:
            correct_predictions += 1
        
            
    accuracy = correct_predictions / total_predictions
    return accuracy, predictions

# Backtest the ARIMA prediction method for 1 day ahead on GOOGL dataset
accuracy_arima_v2, arima_predictions_v2 = backtest_arima_updated_v2(data_googl['Close'].values)
print(f"Accuracy: {accuracy_arima_v2:.2%}")
