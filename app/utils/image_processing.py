from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Modelin giriş boyutuna göre boyutlandırma
    image = np.array(image) / 255.0   # Normalizasyon
    image = np.expand_dims(image, axis=0)  # Model için batch boyutu ekleme
    return image
