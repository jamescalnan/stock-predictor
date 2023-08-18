import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.traceback import install
import logging

FORMAT = "%(asctime)s — %(message)s"
logging.basicConfig(
    level="INFO", 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[RichHandler(rich_tracebacks=True)]
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

logger.info("Loading the dataset...")
data = pd.read_csv('AMZN.csv')
data['Return'] = data['Close'].pct_change()

logger.info("Creating lagged features...")
lag_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for feature in lag_features:
    for i in range(1, 6):
        data[f'{feature}_Lag_{i}'] = data[feature].shift(i)

logger.info("Calculating moving averages...")
data['MA5'] = data['Close'].rolling(window=5).mean()
data['MA30'] = data['Close'].rolling(window=30).mean()

logger.info("Calculating RSI...")
data['RSI'] = compute_rsi(data['Close'])

logger.info("Cleaning up dataset...")
data = data.dropna()

data['Target'] = (data['Return'] > 0).astype(int)

X = data.drop(columns=['Date', 'Return', 'Target'])
y = data['Target']

param_dist = {
    'n_estimators': np.arange(50, 501, 50),
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'max_depth': np.arange(3, 16, 1),
    'min_samples_split': np.arange(2, 11, 1),
    'min_samples_leaf': np.arange(1, 11, 1)
}

clf = GradientBoostingClassifier()

tscv = TimeSeriesSplit(n_splits=5)

logger.info("Starting RandomizedSearchCV...")
search = RandomizedSearchCV(
    clf, 
    param_distributions=param_dist, 
    n_iter=5, 
    scoring='accuracy', 
    n_jobs=-1, 
    cv=tscv, 
    verbose=1, 
    random_state=42
)

search.fit(X, y)

best_params = search.best_params_

logger.info("Performing Time Series Cross-Validation with best parameters...")
clf_best = GradientBoostingClassifier(**best_params)
scores = []
for train_index, test_index in track(tscv.split(X), description="Cross-Validation"):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf_best.fit(X_train, y_train)
    y_pred = clf_best.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

avg_score = np.mean(scores)

logger.info(f"Best Parameters: {best_params}")
logger.info(f"Average Accuracy: {avg_score:.2%}")
