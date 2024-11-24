from fastapi import APIRouter, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

router = APIRouter()

# Eğitilmiş OCR modelinin yüklenmesi
ocr_model = tf.keras.models.load_model("models/weights/ocr_weights.h5")

@router.post("/scan_receipt")
async def scan_receipt(file: UploadFile = File(...)):
    # Görüntüyü yükle ve işle
    image = Image.open(io.BytesIO(await file.read())).convert("L").resize((224, 224))
    input_arr = np.array(image)[np.newaxis, ..., np.newaxis] / 255.0
    
    # Model tahmini
    predictions = ocr_model.predict(input_arr)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return {"success": True, "predicted_class": int(predicted_class)}
