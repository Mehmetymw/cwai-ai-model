
from models.waste_tracking_model import create_waste_tracking_model
from utils.data_loader import load_waste_data
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Eğitim parametreleri
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = "models/weights/waste_tracking_weights.h5"

# Veri yükleme
print("Atık izleme verileri yükleniyor...")
train_data, val_data = load_waste_data(batch_size=BATCH_SIZE)

# Model oluşturma
print("Atık izleme modeli oluşturuluyor...")
model = create_waste_tracking_model()

# Callbacks tanımlama
checkpoint = ModelCheckpoint(
    filepath="models/weights/best_waste_tracking_weights.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

# Modeli eğitme
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping]
)

# Eğitim sonrası en iyi modeli yükleme
model.load_weights("models/weights/best_waste_tracking_weights.h5")

# Model ağırlıklarını kaydetme
model.save(MODEL_SAVE_PATH)
print("Atık izleme modeli başarıyla eğitildi ve kaydedildi.")
