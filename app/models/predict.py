import numpy as np
from tensorflow.keras.models import load_model
from app.utils.image_processing import preprocess_image

model = load_model('models/food_classification_model.h5')

with open('models/classes.txt', 'r') as f:
    classes = [line.strip() for line in f]

calories = [300] * len(classes)  
nutrients = [{"Protein": 10, "Fat": 5, "Carbs": 20}] * len(classes)

def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    if predicted_class >= len(classes):
        raise ValueError(f"Predicted class index {predicted_class} is out of range for available classes: {classes}")

    predicted_food = classes[predicted_class]
    predicted_calories = calories[predicted_class]
    predicted_nutrients = nutrients[predicted_class]

    return {
        "food": predicted_food,
        "calories": predicted_calories,
        "nutrients": predicted_nutrients
    }
