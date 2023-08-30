import pandas as pd
import os, sys
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json

import numpy as np

from rich.console import Console
from rich.logging import RichHandler
import logging

import path
directory = path.Path(__file__).abspath().parent


import matplotlib.pyplot as plt


FORMAT = "%(asctime)s — %(message)s"
logging.basicConfig(
    level="INFO", 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

c = Console()


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
    csv_files = [f for f in os.listdir(f'{directory}/historical_data/') if f.endswith('.csv')]
    
    if not csv_files:
        logger.error("No suitable CSV files found in the current directory.")
        return None

    logger.info("Available CSV files:")
    for idx, file in enumerate(csv_files, start=1):
        logger.info(f"{idx}. {file}")

    choice = int(c.input(f"{' ' * 33}")) - 1

    logger.info(f"Loading the dataset {csv_files[choice]}...")
    return pd.read_csv(f'{directory}/historical_data/{csv_files[choice]}'), csv_files[choice]


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


def GBC_Train(data, ticker, best_params):
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

    # Print the feature importances sorted in descending order
    feature_importances = pd.DataFrame(reg.feature_importances_, index=X_train.columns, columns=['Importance'])
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    c.print('\n')
    logger.info("Feature Importances:")
    for idx, row in feature_importances.iterrows():
        # Have all of the indexes be lined up
        if len(idx) < 25:
            idx += ' ' * (25 - len(idx))

        logger.info(f"{idx}: {row['Importance']:.2%}")


    test_score = np.zeros((best_params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(X_test)):
        test_score[i] = mean_squared_error(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(best_params["n_estimators"]) + 1,
        reg.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(best_params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()


    # from sklearn.inspection import permutation_importance

    # feature_importance = reg.feature_importances_
    # sorted_idx = np.argsort(feature_importance)
    # pos = np.arange(sorted_idx.shape[0]) + 0.5
    # fig = plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.barh(pos, feature_importance[sorted_idx], align="center")
    # plt.yticks(pos, np.array(X_train.columns)[sorted_idx])
    # plt.title("Feature Importance (MDI)")

    # result = permutation_importance(
    #     reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    # )
    # sorted_idx = result.importances_mean.argsort()
    # plt.subplot(1, 2, 2)
    # plt.boxplot(
    #     result.importances[sorted_idx].T,
    #     vert=False,
    #     labels=np.array(X_train.columns)[sorted_idx],
    # )
    # plt.title("Permutation Importance (test set)")
    # fig.tight_layout()
    # plt.show()


    return reg, mse, r2, data

def save_best_params(best_params, ticker):
    # Create a dictionary to hold the best parameters
    best_params_dict = best_params

    # Convert the dictionary to a JSON string
    best_params_json = json.dumps(best_params_dict, indent=4)

    # Define the file name based on the ticker
    file_name = f"gbr_best_params/{ticker}_best_params.json"

    # Write the JSON string to a file
    with open(file_name, "w") as f:
        f.write(best_params_json)

    logger.info(f"Best parameters for {ticker} saved to {file_name}.")

def GBC_Best_Parameters(data, ticker):
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
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Exhausive Grid Search
    # param_grid = {
    #     'n_estimators': np.arange(50, 501, 50),
    #     'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    #     'max_depth': np.arange(3, 16, 1),
    #     'min_samples_split': np.arange(2, 11, 1),
    #     'min_samples_leaf': np.arange(1, 11, 1),
    #     'subsample': [0.5, 0.75, 1]
    # }

    # Non-Exhaustive Grid Search
    param_grid = {
        'n_estimators': [250],
        'learning_rate': [0.1, 0.2],
        'max_depth': [5, 6],
        'min_samples_split': [4, 6],
        'min_samples_leaf': [2, 3],
        'subsample': [1]
    }

    # Initialize GradientBoostingRegressor
    reg = GradientBoostingRegressor(random_state=42)
    
    # Initialize GridSearchCV
    grid = GridSearchCV(estimator=reg, param_grid=param_grid, 
                        scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=3)
    
    # Fit data to GridSearchCV
    grid.fit(X_train, y_train)
    
    # Get the best parameters
    best_params = grid.best_params_
    logger.info(f"Best Parameters: {best_params}")
    
    
    # Save the best parameters to a JSON file
    save_best_params(best_params, ticker)

    return best_params
    
def load_best_params(ticker):
    file_name = f"gbr_best_params/{ticker}_best_params.json"
    logger.info(f"Loading best parameters from {file_name}...")
    try:
        with open(file_name, 'r') as f:
            best_params = json.load(f)
        logger.info(f"Best parameters for {ticker} loaded successfully.")
        return best_params
    except FileNotFoundError:
        return None

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
    data, file_name = get_file_selection()

    # Get the best parameters for the ticker
    # GBC_Best_Parameters(data, file_name.split('.')[0])

    # Load the best parameters for the ticker
    if '_' in file_name:
        ticker = file_name.split('_')[0]
    else:
        ticker = file_name.split('.')[0]
    best_params = load_best_params(ticker)

    if best_params is None:
        logger.error("No best parameters found.")
        logger.info("Would you like to train a new model with the best parameters (y) or use default parameters (n)?")
        choice = c.input(f"{' ' * 33}").lower()

        if choice == 'y':
            best_params =  GBC_Best_Parameters(data, ticker)
        else:
            # Use the default parameters
            best_params = {
                'n_estimators': 250,
                'min_samples_split': 6,
                'min_samples_leaf': 4,
                'max_depth': 3,
                'learning_rate': 0.2,
                'subsample': 1
            }
            logger.info("Using default parameters.")

    c.print()

    
    data['Return'] = data['Close'].pct_change()


    gbc_clf, mse, r2, modified_data = GBC_Train(data, file_name.split('.')[0], best_params)

    results = compare_predictions_with_actual(modified_data, gbc_clf)
    c.print('\n')

    # date = '2023-08-18'

    # # Apply feature engineering
    # with_features = apply_feature_engineering(data, file_name.split('.')[0])

    # X_for_date = with_features[with_features['Date'] == date].drop(columns=['Date'])

    # predicted_close = gbc_clf.predict(X_for_date)

    # # Increment the date by 1 day
    # date = (pd.to_datetime(date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    # # If the date is a weekend, increment it by 1 day until it's a weekday
    # while pd.to_datetime(date).weekday() >= 5:
    #     date = (pd.to_datetime(date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

    # actual = 175.84

    # logger.info(f"Predicted closing price for {date}: {predicted_close[0]:.2f}, actual closing price: {actual}")

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
