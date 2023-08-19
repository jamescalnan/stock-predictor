import pandas as pd
import os, sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

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

def  fetch_news(stock_symbol, date):
    url = f"https://newsapi.org/v2/everything?q={stock_symbol}&from={date}&to={date}&apiKey=ef2e0d24af4f471bbd9d1b34c01d1360"
    response = requests.get(url)
    news_data = response.json()
    
    # Check if the response contains the 'articles' key
    if 'articles' in news_data:
        if news_data['articles']:
            return news_data['articles'][0]['title'] or news_data['articles'][0]['content']
    else:
        # Print out the response to debug any potential issues
        print(f"Unexpected response from News API for {stock_symbol} on {date}: {news_data}")
    return None

# Analyze sentiment of the news article
def analyze_sentiment(news_article):
    analysis = TextBlob(news_article)
    return (analysis.sentiment.polarity + 1) / 2  # Normalize score to [0,1]

def get_sentiment(stock_symbol, date):
    saved_score = sentiment_data[(sentiment_data['Date'] == date) & (sentiment_data['Ticker'] == stock_symbol)]['Score'].values

    if saved_score:
        return saved_score[0]

    news_article = fetch_news(stock_symbol, date)
    if news_article:
        score = analyze_sentiment(news_article)
        # Save the new sentiment score to the file
        with open(SENTIMENT_FILE, 'a') as f:
            f.write(f"{date},{stock_symbol},{score}\n")
        return score
    return None

def GBC_Train(data, ticker):
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

    # data['News_Sentiment'] = [get_sentiment(ticker, date) for date in data['Date']]

    logger.info("Generating interaction and polynomial features...")
    selected_features = ['Open', 'High', 'Low', 'Close']
    for feature in selected_features:
        data[f'{feature}_Squared'] = data[feature] ** 2  # Polynomial feature
        for other_feature in selected_features:
            if feature != other_feature:
                data[f'{feature}_x_{other_feature}'] = data[feature] * data[other_feature]  # Interaction feature

    logger.info("Cleaning up dataset...")
    data = data.copy()
    data = data.dropna()
    data.loc[:, 'Target'] = (data['Return'] > 0).astype(int)


    

    X = data.drop(columns=['Date', 'Return', 'Target'])
    y = data['Target']

    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the parameter grid
    # param_grid = {
    #     'n_estimators': [100, 250, 450],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_depth': [3, 4, 5, 6],
    #     'learning_rate': [0.01, 0.05, 0.1]
    # }

    param_grid = {
        'n_estimators': np.arange(50, 501, 50),
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': np.arange(3, 16, 1),
        'min_samples_split': np.arange(2, 11, 1),
        'min_samples_leaf': np.arange(1, 11, 1)
    }

    # Create a base classifier
    gbc = GradientBoostingClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, 
                               scoring='accuracy', cv=3, n_jobs=-1, verbose=2)

    logger.info("Performing grid search...")
    # with c.status('', spinner='line'):
    grid_search.fit(X_train, y_train)

    # Extract the best parameters from GridSearchCV
    best_params = grid_search.best_params_
    logger.info(f"Best parameters found: {best_params}")

    # Train the model with the best parameters
    clf = GradientBoostingClassifier(**best_params)
    clf.fit(X_train, y_train)

    logger.info("Making predictions on the test set...")
    y_pred = clf.predict(X_test)

    logger.info("Calculating accuracy...")
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2%}")

    return clf, accuracy


# Adding the LSTM_Train function
def LSTM_Train(data, sequence_length=60):
    

    # Data scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create sequences of data
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    X, y = create_dataset(scaled_data, sequence_length)

    # Split data into training and testing sets (80-20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]

    # Reshape data to be [samples, time steps, features] which is required for LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # LSTM model architecture
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1, validation_data=(X_test, y_test))

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Transform back to original scale
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    return model  # Return the trained LSTM model


def backtest_LSTM(data, lstm_model, sequence_length=60):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score

    # Data scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create sequences of data
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    X, y = create_dataset(scaled_data, sequence_length)

    # Reshape data to be [samples, time steps, features] which is required for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Predictions
    y_pred_scaled = lstm_model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # Calculate returns
    actual_returns = data['Close'].iloc[sequence_length + 1:].pct_change().dropna()
    predicted_returns = pd.Series(y_pred.flatten()).pct_change().dropna()

    # Convert returns to binary (1 for increase, 0 for decrease or no change)
    y_actual_binary = (actual_returns > 0).astype(int).values
    y_pred_binary = (predicted_returns > 0).astype(int).values

    # Accuracy
    accuracy = accuracy_score(y_actual_binary, y_pred_binary)

    return accuracy



# Refactoring the main function
def main():
    file_name = get_file_selection()
    if not file_name:
        return

    os.system('cls' if os.name == 'nt' else 'clear')

    logger.info(f"Loading the dataset {file_name}...")
    data = pd.read_csv(file_name)
    data['Return'] = data['Close'].pct_change()
    
    # sequence_length = 20

    gbc_clf, gbc_accuracy = GBC_Train(data, file_name.split('.')[0])
    # lstm_model = LSTM_Train(data, sequence_length)  # Train the LSTM model

    # logger.info("Backtesting the LSTM model...")
    # lstm_accuracy = backtest_LSTM(data, lstm_model, sequence_length)

    # logger.info(f"Gradient Boosting Classifier Accuracy: {gbc_accuracy:.2%}")
    # logger.info(f"LSTM Accuracy: {lstm_accuracy:.2%}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExiting...")
        sys.exit(0)
