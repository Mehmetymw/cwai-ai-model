import cv2
import numpy as np
from scripts.food_detection import detect_food_label
import json

# CO2 ve yoğunluk JSON'u yükleme
with open("data/co2_labels.json", "r") as file:
    co2_labels = json.load(file)

# Piksel farkı hesaplama
def calculate_pixel_difference(before_image_path, after_image_path):
    """
    Yemek öncesi ve sonrası görüntüler arasındaki piksel farkını hesaplar.
    """
    # Görselleri yükleme ve griye dönüştürme
    before_image = cv2.imread(before_image_path, cv2.IMREAD_GRAYSCALE)
    after_image = cv2.imread(after_image_path, cv2.IMREAD_GRAYSCALE)

    # Piksel farkını hesaplama
    pixel_difference = cv2.absdiff(before_image, after_image)
    return np.sum(pixel_difference)

# Atık ve Karbon Ayak İzi Hesaplama
def calculate_waste_and_co2(before_image_path, after_image_path):
    """
    Yemek öncesi ve sonrası arasındaki atık hacmi ve karbon ayak izini hesaplar.
    """
    # Yemek öncesi fotoğraftan gıda label'ını tespit et
    detected_label = detect_food_label(before_image_path)

    # Density ve CO2 değerlerini JSON'dan bul
    food_info = next((item for item in co2_labels if item["class"] == detected_label), None)
    if not food_info:
        raise ValueError(f"Gıda sınıfı '{detected_label}' için CO2 ve yoğunluk bilgisi bulunamadı!")

    density = food_info["density"]
    co2 = food_info["co2"]

    # Piksel farkını hesapla
    pixel_difference = calculate_pixel_difference(before_image_path, after_image_path)

    # Hacim ve karbon ayak izi hesaplama
    waste_volume = pixel_difference / density  # Hacim (ml)
    carbon_footprint = (waste_volume / 1000) * co2  # CO2 (kg)

    return {
        "detected_label": detected_label,
        "waste_volume": waste_volume,
        "carbon_footprint": carbon_footprint
    }
