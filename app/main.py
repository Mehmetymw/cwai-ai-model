from fastapi import FastAPI
from app.api.endpoints import router as api_router

app = FastAPI(
    title="Yemek Tanıma ve Besin Analizi API",
    description="Yemek fotoğrafları üzerinden yemek türü, kalori ve besin değerlerini tahmin eden API",
    version="1.0.0"
)

app.include_router(api_router)
