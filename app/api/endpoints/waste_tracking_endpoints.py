from fastapi import APIRouter, HTTPException, UploadFile, File
from scripts.waste_tracking import calculate_waste_and_co2
import os

router = APIRouter()

@router.post("/waste_tracking")
async def waste_tracking(before_image: UploadFile = File(...), after_image: UploadFile = File(...)):
    """
    Yemek öncesi ve sonrası fotoğraflarını alır, food detection ve waste tracking hesaplamalarını yapar.
    """
    try:
        # Dosyaları geçici olarak kaydet
        before_image_path = f"temp/{before_image.filename}"
        after_image_path = f"temp/{after_image.filename}"

        os.makedirs("temp", exist_ok=True)
        with open(before_image_path, "wb") as f:
            f.write(await before_image.read())
        with open(after_image_path, "wb") as f:
            f.write(await after_image.read())

        # Waste tracking işlemi
        metrics = calculate_waste_and_co2(before_image_path, after_image_path)

        # Geçici dosyaları sil
        os.remove(before_image_path)
        os.remove(after_image_path)

        # Sonuçları döndür
        return {
            "success": True,
            "detected_label": metrics["detected_label"],
            "waste_volume_ml": round(metrics["waste_volume"], 2),
            "carbon_footprint_kg": round(metrics["carbon_footprint"], 2),
        }

    except Exception as e:
        return {"success": False, "error": f"Hata oluştu: {str(e)}"}
