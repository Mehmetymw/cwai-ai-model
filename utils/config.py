import os

class Config:
    OCR_MODEL_PATH = os.getenv("OCR_MODEL_PATH", "models/weights/ocr_weights.h5")
    FOOD_MODEL_PATH = os.getenv("FOOD_MODEL_PATH", "models/weights/food_weights.h5")
    CALORIE_MODEL_PATH = os.getenv("CALORIE_MODEL_PATH", "models/weights/calorie_weights.h5")
