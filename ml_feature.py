import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

# Load the dataset
data_googl = pd.read_csv('AMZN.csv')
data_googl['Return'] = data_googl['Close'].pct_change()

# Creating lagged features for the given columns
lag_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for feature in lag_features:
    for i in range(1, 6):  # Creating 5 lagged values
        data_googl[f'{feature}_Lag_{i}'] = data_googl[feature].shift(i)

# Adding moving averages
data_googl['MA5'] = data_googl['Close'].rolling(window=5).mean()
data_googl['MA30'] = data_googl['Close'].rolling(window=30).mean()

# Adding RSI
data_googl['RSI'] = compute_rsi(data_googl['Close'])

# Drop NA values after feature engineering
data_googl = data_googl.dropna()

# Keeping the same target variable: Whether the stock went up (1) or down (0) the next day
data_googl['Target'] = (data_googl['Return'] > 0).astype(int)

# Create new features (X) and target (y) datasets
X_new = data_googl.drop(columns=['Date', 'Return', 'Target'])
y_new = data_googl['Target']

# Splitting data into training and testing sets (80% train, 20% test)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Classifier with the new features
clf_new = GradientBoostingClassifier()
clf_new.fit(X_train_new, y_train_new)

# Predict on the test set
y_pred_new = clf_new.predict(X_test_new)

# Calculate accuracy with the new features
accuracy_ml_new = accuracy_score(y_test_new, y_pred_new)
print(f"Accuracy: {accuracy_ml_new:.2%}")
