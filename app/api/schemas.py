from pydantic import BaseModel

class PredictionResponse(BaseModel):
    food: str
    calories: int
    nutrients: dict
