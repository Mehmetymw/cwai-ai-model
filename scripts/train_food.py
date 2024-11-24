from models.food_model import create_food_classification_model
from utils.data_loader import load_food_data

# Model oluşturma
model = create_food_classification_model()

# Eğitim ve doğrulama verilerini yükleme
train_data, val_data = load_food_data()

# Modeli eğitme
model.fit(train_data, epochs=20, validation_data=val_data)

# Model ağırlıklarını kaydetme
model.save("models/weights/food_weights.h5")
print("Gıda tanıma modeli başarıyla eğitildi ve kaydedildi.")
