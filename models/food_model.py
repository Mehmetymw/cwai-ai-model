import tensorflow as tf
from tensorflow.keras import layers, models

def create_food_classification_model(input_shape=(224, 224, 3), num_classes=101):
    # Gıda tanıma için önceden eğitilmiş bir modelin temelini kullanıyoruz (örneğin ResNet50)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False  # Önceden eğitilmiş katmanları sabit tut
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
