import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import Sequence
from sklearn.preprocessing import MinMaxScaler
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
c = Console()

logger.info('Reading data...')
# Load data
data = pd.read_csv('TSLA.csv')
data['Return'] = data['Close'].pct_change().fillna(0)

logger.info('Calculating drift and volatility...')
# Calculate drift and volatility for GBM
drift = data['Return'].mean()
volatility = data['Return'].std()

def gbm_simulation(S0, drift, volatility, days=30, simulations=1000):
    """Simulates stock prices using Geometric Brownian Motion."""
    dt = 1  # time increment (1 day)
    simulated_prices = np.zeros((days, simulations))
    simulated_prices[0] = S0
    for day in range(1, days):
        dS = simulated_prices[day-1] * (drift * dt + volatility * np.random.normal(0, 1, simulations) * np.sqrt(dt))
        simulated_prices[day] = simulated_prices[day-1] + dS
    return simulated_prices

logger.info('Simulating paths...')
# Simulate future paths using GBM
forecast_days = 30
all_simulated_paths = []
for price in data['Close']:
    simulated_paths = gbm_simulation(price, drift, volatility, days=forecast_days)
    all_simulated_paths.extend(simulated_paths.T)

logger.info('Preparing data...')
# Combine real data with simulated paths
combined_data = np.concatenate([data['Close'].values, np.array(all_simulated_paths).flatten()])

logger.info('Preparing sequences...')
# Prepare sequences for LSTM
sequence_length = 10
X, y = [], []
for i in track(range(len(combined_data) - sequence_length - 1), description=f'{" " * 32}', transient=False):
    sequence = combined_data[i:i + sequence_length]
    target = combined_data[i + sequence_length]
    X.append(sequence)
    y.append(1 if target > sequence[-1] else 0)

X = np.array(X)
y = np.array(y)

# Reshape X for LSTM [samples, time steps, features]
with c.status('', spinner='line'):
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

logger.info('Splitting data...')
# Splitting data into train and test sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

logger.info('Normalizing data...')
# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Define TimeSeriesGenerator
class TimeSeriesGenerator(Sequence):
    def __init__(self, X_data, y_data, batch_size, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X_data))
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.take(self.X_data, indices, axis=0)
        y = np.take(self.y_data, indices, axis=0)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Create data generators
batch_size = 32
train_generator = TimeSeriesGenerator(X_train, y_train, batch_size)
test_generator = TimeSeriesGenerator(X_test, y_test, batch_size)

logger.info('Building model...')
# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

logger.info('Training model...')
# Train the model using data generators
model.fit(train_generator, epochs=5, validation_data=test_generator)

logger.info('Predicting...')
# Predict on the test set
y_pred = model.predict(X_test)
y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]

logger.info('Evaluating...')
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")
