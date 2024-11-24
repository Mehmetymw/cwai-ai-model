from fastapi import APIRouter, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

router = APIRouter()

# Eğitilmiş kalori hesaplama modelinin yüklenmesi
calorie_model = tf.keras.models.load_model("models/weights/calorie_weights.h5")

@router.post("/calculate_calories")
async def calculate_calories(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    input_arr = np.array(image)[np.newaxis, ...] / 255.0
    
    calorie_prediction = calorie_model.predict(input_arr)[0][0]

    return {
        "success": True,
        "calories_per_serving": float(calorie_prediction)
    }
