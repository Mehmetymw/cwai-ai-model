from fastapi import FastAPI
from app.api.endpoints import ocr_endpoints, food_endpoints, calorie_endpoints

app = FastAPI(
    title="CookWiseAI Image Recognition API",
    description="API for OCR, Food Recognition, and Calorie Calculation",
    version="1.0.0",
)

app.include_router(ocr_endpoints.router, prefix="/ocr", tags=["OCR"])
app.include_router(food_endpoints.router, prefix="/food", tags=["Food Recognition"])
app.include_router(calorie_endpoints.router, prefix="/calories", tags=["Calorie Calculation"])
