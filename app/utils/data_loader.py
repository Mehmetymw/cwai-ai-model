import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    data = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    return data
