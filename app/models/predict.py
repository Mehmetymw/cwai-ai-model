import numpy as np
from tensorflow.keras.models import load_model
from app.utils.image_processing import preprocess_image

model = load_model('models/food_classification_model.h5')

def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    classes = ["Pizza", "Salad", "Burger"]
    calories = [300, 150, 500]
    nutrients = [
        {"Protein": 12, "Fat": 10, "Carbs": 30},
        {"Protein": 5, "Fat": 2, "Carbs": 15},
        {"Protein": 20, "Fat": 25, "Carbs": 50}
    ]

    predicted_food = classes[predicted_class]
    predicted_calories = calories[predicted_class]
    predicted_nutrients = nutrients[predicted_class]

    return {
        "food": predicted_food,
        "calories": predicted_calories,
        "nutrients": predicted_nutrients
    }
