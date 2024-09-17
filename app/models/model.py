from tensorflow.keras.models import load_model
from app.config import Config

def load_trained_model():
    model = load_model(Config.MODEL_PATH)
    print(f"Model {Config.MODEL_PATH} yüklendi.")
    return model

model = load_trained_model()
