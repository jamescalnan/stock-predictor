import os
import sys
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from rich.console import Console
from rich.logging import RichHandler
import logging

c = Console()

FORMAT = "%(asctime)s — %(message)s"
logging.basicConfig(
    level="INFO", 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[RichHandler()]
)

logger = logging.getLogger("rich")

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
    sys.stdout.write("\033[F")
    return csv_files[choice]

def compute_rsi(data, window=14):
    delta = data.diff()
    loss = delta.where(delta < 0, 0)
    gain = -delta.where(delta > 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def compute_features(data):
    data['Return'] = data['Close'].pct_change()
    
    lag_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for feature in lag_features:
        for i in range(1, 6):
            data[f'{feature}_Lag_{i}'] = data[feature].shift(i)

    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA30'] = data['Close'].rolling(window=30).mean()
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['RSI'] = compute_rsi(data['Close'])

    for i in range(1, 6):
        data[f'RSI_Lag_{i}'] = data['RSI'].shift(i)

    return data

def main():
    apple_file = get_file_selection()
    msft_file = get_file_selection()

    apple_data = pd.read_csv(apple_file)
    msft_data = pd.read_csv(msft_file)

    apple_data = apple_data[apple_data['Date'].isin(msft_data['Date'])]
    msft_data = msft_data[msft_data['Date'].isin(apple_data['Date'])]

    apple_data = compute_features(apple_data)
    msft_data = compute_features(msft_data)

    apple_data['Relative_Performance'] = apple_data['Return'] - msft_data['Return']

    combined_data = apple_data.merge(msft_data, on='Date', suffixes=('', '_MSFT'))

    combined_data = combined_data.dropna()
    combined_data['Target'] = (combined_data['Return'] > 0).astype(int)

    X = combined_data.drop(columns=['Date', 'Return', 'Target'])
    y = combined_data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = {
        'n_estimators': 450,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_depth': 6,
        'learning_rate': 0.1
    }

    clf = GradientBoostingClassifier(**best_params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
