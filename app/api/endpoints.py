from fastapi import APIRouter, File, UploadFile
from PIL import Image
from app.models.predict import predict

router = APIRouter()

@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image = Image.open(file.file)
    result = predict(image)
    return result
