
from fastapi import APIRouter, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

router = APIRouter()

try:
    calorie_model = tf.keras.models.load_model("models/weights/calorie_weights.h5")
except Exception as e:
    raise RuntimeError(f"Model yüklenirken hata oluştu: {e}")

@router.post("/calculate_calories")
async def calculate_calories(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((224, 224))
        input_arr = np.array(image)[np.newaxis, ...] / 255.0

        calorie_prediction = calorie_model.predict(input_arr)[0][0]

        return {
            "success": True,
            "calories": float(calorie_prediction)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
