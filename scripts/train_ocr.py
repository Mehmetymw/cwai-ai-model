
import tensorflow as tf
from models.ocr_model import create_ocr_model
from utils.data_loader import load_ocr_data
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Model ve eğitim parametreleri
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50  # Epoch sayısı artırıldı
MODEL_SAVE_PATH = "models/weights/ocr_weights.h5"

print("Veri yükleniyor...")
train_data, val_data = load_ocr_data(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# Model oluşturma
print("OCR modeli oluşturuluyor...")
ocr_model = create_ocr_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=26)

# Callbacks tanımlama
checkpoint = ModelCheckpoint(
    filepath="models/weights/best_ocr_weights.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

# Modeli eğitme
history = ocr_model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# Eğitim sonrası en iyi modeli yükleme
ocr_model.load_weights("models/weights/best_ocr_weights.h5")

# Model ağırlıklarını kaydetme
ocr_model.save(MODEL_SAVE_PATH)
print("OCR modeli başarıyla eğitildi ve kaydedildi.")
