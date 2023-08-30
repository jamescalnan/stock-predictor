import pandas as pd
import os, sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

from rich.console import Console
from rich.logging import RichHandler
import logging



FORMAT = "%(asctime)s — %(message)s"
logging.basicConfig(
    level="INFO", 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[RichHandler()]
)

logger = logging.getLogger("rich")
import requests
from textblob import TextBlob

c = Console()
SENTIMENT_FILE = "sentiment_scores.csv"

# Check if sentiment file exists, if not create it
if not os.path.exists(SENTIMENT_FILE):
    with open(SENTIMENT_FILE, 'w') as f:
        f.write("Date,Ticker,Score\n")

# Load existing sentiment scores
sentiment_data = pd.read_csv(SENTIMENT_FILE)

def compute_rsi(data, window=14):
    """Compute the RSI (Relative Strength Index) of the data."""
    delta = data.diff()
    loss = delta.where(delta < 0, 0)
    gain = -delta.where(delta > 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_file_selection():
    """Returns the selected file from a menu of available CSV files."""
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    
    if not csv_files:
        logger.error("No suitable CSV files found in the current directory.")
        return None

    logger.info("Available CSV files:")
    for idx, file in enumerate(csv_files, start=1):
        logger.info(f"{idx}. {file}")

    choice = int(c.input(f"{' ' * 33}")) - 1
    # sys.stdout.write("\033[F")
    return csv_files[choice]


def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    """Compute the MACD (Moving Average Convergence Divergence) of the data."""
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(data, window=20, num_std_dev=2):
    """Compute Bollinger Bands."""
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def compute_stochastic_oscillator(data, window=14):
    """Compute Stochastic Oscillator."""
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
    return k


def apply_feature_engineering(data, ticker):
    data = data.copy()
    lag_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for feature in lag_features:
        for i in range(1, 6):  # Creating 5 lagged values
            data[f'{feature}_Lag_{i}'] = data[feature].shift(i)

    logger.info("Calculating moving averages...")
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA30'] = data['Close'].rolling(window=30).mean()
    data['MA7'] = data['Close'].rolling(window=7).mean()  # 7-day MA

    logger.info("Calculating RSI...")
    data['RSI'] = compute_rsi(data['Close'])

    # Create lagged features for RSI
    for i in range(1, 6): 
        data[f'RSI_Lag_{i}'] = data['RSI'].shift(i)

    logger.info("Calculating MACD and Signal Line...")
    data['MACD'], data['Signal_Line'] = compute_macd(data)

    logger.info("Calculating Bollinger Bands...")
    data['Upper_Band'], data['Lower_Band'] = compute_bollinger_bands(data)

    logger.info("Calculating Stochastic Oscillator...")
    data['Stochastic_Oscillator'] = compute_stochastic_oscillator(data)

    # data['News_Sentiment'] = [get_sentiment(ticker, date) for date in data['Date']]

    # News Sentiment
    if '_' in ticker:
        ticker = ticker.split('_')[0]

    news_sentiment = pd.read_csv(f'news_sentiment_data/{ticker}_news_data.csv')

    # Assign column names to the data in the form date, $ticker, sentiment_score, url, title
    news_sentiment.columns = ['Date', 'Ticker', 'Score', 'URL', 'Title']

    # Index the data by date
    news_sentiment = news_sentiment.set_index('Date')

    # Need to add these sentiment scores to the data, need to loop through the sentiment data 
    # and if there is a score for that date then add it to the data
    # Loop through the data and check if the date is in the sentiment data
    # If it is then add the score to the data
    # If not then add a 0

    # Create a new column in the data called sentiment score
    data['Sentiment_Score'] = 0

    # Loop through the data
    for index, row in data.iterrows():
        date = row['Date']
        
        # Check if the index is in the sentiment data
        if date in news_sentiment.index:
            # If it is then add the score to the data
            # Need to check if there is more than one score for that date, if there is then take the average
            # Get the scores for that date
            scores = news_sentiment.loc[date]['Score']
            # Check if there is more than one score
            if not type(scores) == np.float64 and len(scores) > 1:
                # If there is then take the average
                data.at[index, 'Sentiment_Score'] = np.mean(scores)
            else:
                data.at[index, 'Sentiment_Score'] = news_sentiment.at[date, 'Score']
        # If not then add a 0
        else:
            data.at[index, 'Sentiment_Score'] = 0

    logger.info("Generating interaction and polynomial features...")
    selected_features = ['Open', 'High', 'Low', 'Close']
    for feature in selected_features:
        data[f'{feature}_Squared'] = data[feature] ** 2  # Polynomial feature
        for other_feature in selected_features:
            if feature != other_feature:
                data[f'{feature}_x_{other_feature}'] = data[feature] * data[other_feature]  # Interaction feature

    logger.info("Cleaning up dataset...")
    data = data.dropna()

    return data


def GBC_Train(data, ticker):
    data = data.copy()
    logger.info("Creating lagged features...")
    lag_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for feature in lag_features:
        for i in range(1, 6):  # Creating 5 lagged values
            data[f'{feature}_Lag_{i}'] = data[feature].shift(i)

    logger.info("Calculating moving averages...")
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA30'] = data['Close'].rolling(window=30).mean()
    data['MA7'] = data['Close'].rolling(window=7).mean()  # 7-day MA

    logger.info("Calculating RSI...")
    data['RSI'] = compute_rsi(data['Close'])

    # Create lagged features for RSI
    for i in range(1, 6): 
        data[f'RSI_Lag_{i}'] = data['RSI'].shift(i)

    logger.info("Calculating MACD and Signal Line...")
    data['MACD'], data['Signal_Line'] = compute_macd(data)

    logger.info("Calculating Bollinger Bands...")
    data['Upper_Band'], data['Lower_Band'] = compute_bollinger_bands(data)

    logger.info("Calculating Stochastic Oscillator...")
    data['Stochastic_Oscillator'] = compute_stochastic_oscillator(data)

    # News Sentiment
    if '_' in ticker:
        ticker = ticker.split('_')[0]

    news_sentiment = pd.read_csv(f'news_sentiment_data/{ticker}_news_data.csv')

    # Assign column names to the data in the form date, $ticker, sentiment_score, url, title
    news_sentiment.columns = ['Date', 'Ticker', 'Score', 'URL', 'Title']

    # Index the data by date
    news_sentiment = news_sentiment.set_index('Date')

    # Need to add these sentiment scores to the data, need to loop through the sentiment data 
    # and if there is a score for that date then add it to the data
    # Loop through the data and check if the date is in the sentiment data
    # If it is then add the score to the data
    # If not then add a 0

    # Create a new column in the data called sentiment score
    data['Sentiment_Score'] = 0

    # Loop through the data
    for index, row in data.iterrows():
        date = row['Date']
        
        # Check if the index is in the sentiment data
        if date in news_sentiment.index:
            # If it is then add the score to the data
            # Need to check if there is more than one score for that date, if there is then take the average
            # Get the scores for that date
            scores = news_sentiment.loc[date]['Score']
            # Check if there is more than one score
            if not type(scores) == np.float64 and len(scores) > 1:
                # If there is then take the average
                data.at[index, 'Sentiment_Score'] = np.mean(scores)
            else:
                data.at[index, 'Sentiment_Score'] = news_sentiment.at[date, 'Score']
        # If not then add a 0
        else:
            data.at[index, 'Sentiment_Score'] = 0


    logger.info("Generating interaction and polynomial features...")
    selected_features = ['Open', 'High', 'Low', 'Close']
    for feature in selected_features:
        data[f'{feature}_Squared'] = data[feature] ** 2  # Polynomial feature
        for other_feature in selected_features:
            if feature != other_feature:
                data[f'{feature}_x_{other_feature}'] = data[feature] * data[other_feature]  # Interaction feature

    logger.info("Cleaning up dataset...")
    data = data.dropna()

    # Predict the next day's closing price (so we use a shift of -1)
    data.loc[:, 'Next_Close'] = data['Close'].shift(-1)

    # Drop rows with NaN in 'Next_Close' column
    data = data.dropna(subset=['Next_Close'])

    X = data.drop(columns=['Date', 'Next_Close'])
    y = data['Next_Close']

    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    # param_grid = {
    #     'n_estimators': np.arange(50, 501, 50),
    #     'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    #     'max_depth': np.arange(3, 16, 1),
    #     'min_samples_split': np.arange(2, 11, 1),
    #     'min_samples_leaf': np.arange(1, 11, 1),
    #     'subsample': [0.5, 0.75, 1]
    # }

    # clf = GradientBoostingClassifier()

    # tscv = TimeSeriesSplit(n_splits=5)

    # logger.info("Starting RandomizedSearchCV...")
    # search = RandomizedSearchCV(
    #     clf, 
    #     param_distributions=param_dist, 
    #     n_iter=5, 
    #     scoring='accuracy', 
    #     n_jobs=-1, 
    #     cv=tscv, 
    #     verbose=1, 
    #     random_state=42
    # )

    # search.fit(X, y)

    # best_params = search.best_params_

    best_params = {
        'n_estimators': 250,
        'min_samples_split': 6,
        'min_samples_leaf': 4,
        'max_depth': 3,
        'learning_rate': 0.2,
        'subsample': 1
    }

    # Train the model with the best parameters
    reg = GradientBoostingRegressor(**best_params)
    reg.fit(X_train, y_train)

    logger.info("Making predictions on the test set...")
    y_pred = reg.predict(X_test)

    logger.info("Calculating performance metrics...")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    
    logger.info(f"Mean Squared Error: {mse:.2f}")
    logger.info(f"R^2 Score: {r2:.2%}")

    return reg, mse, r2, data



def predict_next_day_closing(data, model, given_date):
    """
    Predicts the next day's closing price for a given date.

    Parameters:
    - data: The dataset containing the historical stock data.
    - model: The trained GradientBoostingRegressor model.
    - given_date: The date for which we want to predict the next day's closing price.

    Returns:
    - Predicted closing price for the day after the given_date.
    """
    data.loc[:, 'Next_Close'] = data['Close'].shift(-1)
    # Ensure the given_date exists in the dataset
    if given_date not in data['Date'].values:
        raise ValueError(f"Given date {given_date} not found in the dataset.")
    
    # Extract the features for the given_date
    X_for_date = data[data['Date'] == given_date].drop(columns=['Date', 'Next_Close'])
    
    # Save the X_for_date DataFrame to a CSV file
    X_for_date.to_csv('X_for_date.csv', index=False)


    # Predict the closing price for the next day
    predicted_close = model.predict(X_for_date)
    
    return predicted_close[0]

def compare_predictions_with_actual(data, model):
    """
    Loops through all dates in the dataset and compares the predicted closing price 
    with the actual closing price of the next day.

    Parameters:
    - data: The dataset containing the historical stock data.
    - model: The trained GradientBoostingRegressor model.

    Returns:
    - A DataFrame containing the date, predicted closing price, and actual closing price.
    """
    data['Next_Close'] = data['Close'].shift(-1)

    results = []

    # Exclude the last date since we won't have the actual closing price for the day after the last date
    dates_to_predict = data['Date'].iloc[:-1].values
    misses = 0
    accuracy_requirement = 0.001
    prev = None
    for date in dates_to_predict:
        # Extract the features for the current date
        X_for_date = data[data['Date'] == date].drop(columns=['Date', 'Next_Close'])
        
        # Skip the prediction if the feature set contains NaN values
        if X_for_date.isnull().values.any():
            continue
        
        # Predict the closing price for the next day
        predicted_close = model.predict(X_for_date)
        
        # Get the actual closing price for the next day
        actual_close = data[data['Date'] == date]['Next_Close'].values[0]
        
        results.append({
            'Date': date,
            'Actual_Close': actual_close,
            'Predicted_Close': predicted_close[0]
        })

        diff = predicted_close[0] - actual_close

        # Compare theactual close and the predicted close, and if the predicted value predicts a price increase or decrease that is correct then print this
        if prev is not None:
            if (predicted_close[0] > prev and actual_close > prev) or (predicted_close[0] < prev and actual_close < prev):
                logger.info(f"Date: {date}, Correctly predicted a {'rise' if actual_close > prev else 'fall'} from previous close")
            else:
                logger.error(f"Date: {date}, Failed to predict a {'rise' if actual_close > prev else 'fall'} from previous close")


        if outside_1_percent(predicted_close[0], actual_close, accuracy_requirement) :
            logger.error(f"Date: {date}, Actual: ${actual_close:.2f}, Predicted: ${predicted_close[0]:.2f}, Difference: ${diff:.2f}")
            misses += 1
        else:
            logger.info(f"Date: {date}, Actual: ${actual_close:.2f}, Predicted: ${predicted_close[0]:.2f}, Difference: ${diff:.2f}")
        c.print()
        prev = actual_close

    logger.info(f"Misses: {misses}, Total: {len(results)}, Accuracy: {(len(results) - misses) / len(results):.2%}, Accuracy Requirement: {accuracy_requirement*100}%")

    return results

def outside_1_percent(predicted, actual, pct=0.01):
    return abs(predicted - actual) >= (actual * pct)

# Refactoring the main function
def main():
    file_name = get_file_selection()
    if not file_name:
        return

    # os.system('cls' if os.name == 'nt' else 'clear')

    logger.info(f"Loading the dataset {file_name}...")
    data = pd.read_csv(file_name)
    data['Return'] = data['Close'].pct_change()


    gbc_clf, mse, r2, modified_data = GBC_Train(data, file_name.split('.')[0])


    results = compare_predictions_with_actual(modified_data, gbc_clf)
    c.print('\n\n')

    date = '2023-08-18'

    # Apply feature engineering
    with_features = apply_feature_engineering(data, file_name.split('.')[0])

    X_for_date = with_features[with_features['Date'] == date].drop(columns=['Date'])

    predicted_close = gbc_clf.predict(X_for_date)

    # Increment the date by 1 day
    date = (pd.to_datetime(date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    # If the date is a weekend, increment it by 1 day until it's a weekday
    while pd.to_datetime(date).weekday() >= 5:
        date = (pd.to_datetime(date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

    actual = 175.84

    logger.info(f"Predicted closing price for {date}: {predicted_close[0]:.2f}, actual closing price: {actual}")

    # Show the r2 and mse
    logger.info(f"Mean Squared Error: {mse:.2f}")
    logger.info(f"R^2 Score: {r2:.2%}")
    # display the ticker
    logger.info(f"File: {file_name.split('.')[0]}")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExiting...")
        sys.exit(0)
