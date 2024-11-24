from fastapi import APIRouter, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

router = APIRouter()

# Eğitilmiş gıda tanıma modelinin yüklenmesi
food_model = tf.keras.models.load_model("models/weights/food_weights.h5")

@router.post("/identify_ingredients")
async def identify_ingredients(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    input_arr = np.array(image)[np.newaxis, ...] / 255.0
    
    predictions = food_model.predict(input_arr)
    predicted_classes = np.argsort(predictions[0])[-5:]  # En olası 5 sonucu döndür
    
    return {"success": True, "predicted_ingredients": predicted_classes.tolist()}
