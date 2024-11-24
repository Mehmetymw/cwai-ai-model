from scripts.food_detection import detect_food_label
import cv2
import numpy as np
import json

# JSON'dan CO2 ve yoğunluk bilgilerini yükleme
with open("data/co2_labels.json", "r") as file:
    co2_labels = json.load(file)

# Piksel farkı hesaplama
def calculate_pixel_difference(before_image_path, after_image_path):
    """
    Yemek öncesi ve sonrası görüntüler arasındaki piksel farkını hesaplar.
    """
    before_image = cv2.imread(before_image_path, cv2.IMREAD_GRAYSCALE)
    after_image = cv2.imread(after_image_path, cv2.IMREAD_GRAYSCALE)

    # Piksel farkını hesapla
    pixel_difference = cv2.absdiff(before_image, after_image)
    return np.sum(pixel_difference)

# Waste Tracking işlemi
def calculate_waste_and_co2(before_image_path, after_image_path):
    """
    Yemek öncesi ve sonrası arasındaki atık miktarını ve karbon ayak izini hesaplar.
    """
    # Yemek öncesi fotoğraftan besin tespiti
    detected_label = detect_food_label(before_image_path)

    # JSON'dan ilgili besin bilgilerini al
    food_info = next((item for item in co2_labels if item["class"] == detected_label), None)
    if not food_info:
        raise ValueError(f"Gıda sınıfı '{detected_label}' için bilgi bulunamadı!")

    density = food_info["density"]
    co2 = food_info["co2"]

    # Piksel farkını hesapla
    pixel_difference = calculate_pixel_difference(before_image_path, after_image_path)

    # Hacimsel atık ve karbon ayak izi hesaplama
    waste_volume = pixel_difference / density  # ml cinsinden hacim
    carbon_footprint = (waste_volume / 1000) * co2  # kg CO2 cinsinden karbon ayak izi

    return {
        "detected_label": detected_label,
        "waste_volume": waste_volume,
        "carbon_footprint": carbon_footprint
    }
