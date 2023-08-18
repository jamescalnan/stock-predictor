import pandas as pd
import os, sys
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
    sys.stdout.write("\033[F")
    return csv_files[choice]

def main():
    file_name = get_file_selection()
    if not file_name:
        return

    logger.info(f"Loading the dataset {file_name}...")
    data = pd.read_csv(file_name)
    data['Return'] = data['Close'].pct_change()

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

    logger.info("Cleaning up dataset...")
    data = data.dropna()
    data['Target'] = (data['Return'] > 0).astype(int)

    X = data.drop(columns=['Date', 'Return', 'Target'])
    y = data['Target']

    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_params = {
        'n_estimators': 450,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_depth': 6,
        'learning_rate': 0.1
    }

    logger.info("Training the Gradient Boosting Classifier with best parameters...")
    with c.status('', spinner='line'):
        clf = GradientBoostingClassifier(**best_params)
        clf.fit(X_train, y_train)

    logger.info("Making predictions on the test set...")
    y_pred = clf.predict(X_test)

    logger.info("Calculating accuracy...")
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExiting...")
        sys.exit(0)
