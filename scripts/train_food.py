from models.food_model import create_food_classification_model
from utils.data_loader import load_food_data
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Model oluşturma
model = create_food_classification_model()

# Eğitim ve doğrulama verilerini yükleme
train_data, val_data = load_food_data(batch_size=32, img_size=(224, 224))

# Callbacks tanımlama
checkpoint = ModelCheckpoint(
    filepath="models/weights/best_food_weights.keras",  # Yeni uzantı .keras
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1
)

# Modeli eğitme
history = model.fit(
    train_data,
    epochs=1,
    validation_data=val_data,
    callbacks=[checkpoint, early_stopping]
)

# Model ağırlıklarını kaydetme
model.save("models/weights/food_weights.h5")
print("Gıda tanıma modeli başarıyla eğitildi ve kaydedildi.")
