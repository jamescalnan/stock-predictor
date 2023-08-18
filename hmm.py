from hmmlearn import hmm
import pandas as pd
import numpy as np

# Load the data
data_googl = pd.read_csv('GOOGL.csv')
data_googl['Return'] = data_googl['Close'].pct_change().fillna(0)
returns = data_googl['Return'].values[1:].reshape(-1, 1)

# Create and train the HMM with 4 hidden states
model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
model.fit(returns)

def backtest_hmm_updated(model, data):
    """
    Backtests the HMM prediction method over the entire dataset.
    """
    correct_predictions = 0
    total_predictions = len(data) - 1
    for i in range(total_predictions):
        observed_data = data[i].reshape(-1, 1)
        
        # Predict the latent states for the observed data
        hidden_states = model.predict(observed_data)
        
        # Predict the next latent state
        next_state_probs = model.transmat_[hidden_states[-1], :]
        next_state = np.argmax(next_state_probs)
        
        # Predict the next day's return using the mean of the Gaussian distribution for the next state
        predicted_return = model.means_[next_state][0]
        
        predicted_direction = "Increase" if predicted_return > 0 else "Decrease"
        actual_direction = "Increase" if data[i + 1] > 0 else "Decrease"

        print(predicted_direction, actual_direction)
        
        if predicted_direction == actual_direction:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy

# Backtest the HMM prediction method for 1 day ahead on GOOGL dataset
accuracy_hmm = backtest_hmm_updated(model, returns)
print(f"Accuracy: {accuracy_hmm:.2%}")
