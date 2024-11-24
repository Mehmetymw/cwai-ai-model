
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM

def create_waste_tracking_model(input_shape=(10, 3)):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, activation='relu', return_sequences=True),
        LSTM(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # Output: predicted waste amount
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
