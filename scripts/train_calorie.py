from models.calorie_model import create_calorie_prediction_model
from utils.data_loader import load_calorie_data

# Model oluşturma
model = create_calorie_prediction_model()

# Eğitim ve doğrulama verilerini yükleme
train_data, val_data = load_calorie_data()

# Modeli eğitme
model.fit(train_data, epochs=20, validation_data=val_data)

# Model ağırlıklarını kaydetme
model.save("models/weights/calorie_weights.h5")
print("Kalori tahmin modeli başarıyla eğitildi ve kaydedildi.")
