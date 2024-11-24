from pydantic import BaseModel
from typing import List, Dict

class OCRResponse(BaseModel):
    success: bool
    predicted_class: int

class FoodRecognitionResponse(BaseModel):
    success: bool
    predicted_ingredients: List[int]

class CalorieResponse(BaseModel):
    success: bool
    calories_per_serving: float
