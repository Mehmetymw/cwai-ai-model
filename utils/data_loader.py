import tensorflow as tf

# OCR veri yükleyici
def load_ocr_data(batch_size=32, img_size=(224, 224)):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/ocr/train",
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/ocr/val",
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
    )
    return train_ds, val_ds

# Gıda veri yükleyici
def load_food_data(batch_size=32, img_size=(224, 224)):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/food/train",
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=img_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/food/val",
        labels="inferred",
        label_mode="categorical",
        batch_size=batch_size,
        image_size=img_size,
    )
    return train_ds, val_ds

# Kalori veri yükleyici
def load_calorie_data(batch_size=32, img_size=(224, 224)):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/calories/train",
        labels="inferred",
        label_mode="int",  # Kalori tahmini regresyon çıktısı
        batch_size=batch_size,
        image_size=img_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/calories/val",
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
    )
    return train_ds, val_ds
