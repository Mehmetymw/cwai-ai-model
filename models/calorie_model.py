import tensorflow as tf
from tensorflow.keras import layers, models

def create_calorie_prediction_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1)  # Kalori tahmini için tek bir çıktımız var
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model
    