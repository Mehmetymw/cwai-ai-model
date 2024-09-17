import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def train_model(data_dir, model_path):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )

    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(train_data.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=1,
        steps_per_epoch=train_data.samples // train_data.batch_size,
        validation_steps=val_data.samples // val_data.batch_size
    )

    # Modeli kaydetme
    model.save(model_path)

    # Sınıf isimlerini kaydetme
    classes = list(train_data.class_indices.keys())
    with open('models/classes.txt', 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")

    print(f"Model {model_path} konumuna kaydedildi.")
    print("Sınıf isimleri kaydedildi:", classes)

# Eğitim fonksiyonunu çağırma
train_model("data/raw/food-101/images", "models/food_classification_model.h5")
