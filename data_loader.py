import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def load_data(csv_path, sequence_length=20, split_ratios=(0.7, 0.15, 0.15)):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    
    # Add time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Select features (now 4 features instead of 2)
    df = df[['cpu_usage', 'memory_usage', 'hour_of_day', 'day_of_week']].copy()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length][0]  # Still predicting CPU usage (first column)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_data, sequence_length)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    total = len(X)
    train_end = int(split_ratios[0] * total)
    val_end = train_end + int(split_ratios[1] * total)

    X_train, y_train = X_tensor[:train_end], y_tensor[:train_end]
    X_val, y_val = X_tensor[train_end:val_end], y_tensor[train_end:val_end]
    X_test, y_test = X_tensor[val_end:], y_tensor[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
