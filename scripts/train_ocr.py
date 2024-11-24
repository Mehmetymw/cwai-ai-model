import tensorflow as tf
from models.ocr_model import create_ocr_model
from utils.data_loader import load_ocr_data
import os


# Model ve eğitim parametreleri
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20
MODEL_SAVE_PATH = "models/weights/ocr_weights.h5"

print("Veri yükleniyor...")
train_data, val_data = load_ocr_data(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# Model oluşturma
print("OCR modeli oluşturuluyor...")
ocr_model = create_ocr_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=26)

# Eğitim
print("Model eğitiliyor...")
history = ocr_model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    verbose=1
)

# Model ağırlıklarını kaydetme
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
ocr_model.save(MODEL_SAVE_PATH)
print(f"Model başarıyla eğitildi ve {MODEL_SAVE_PATH} konumuna kaydedildi.")
