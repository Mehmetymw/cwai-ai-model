import os
from dotenv import load_dotenv

load_dotenv() 

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "models/food_classification_model.h5")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
