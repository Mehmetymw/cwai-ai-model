import os
import tensorflow as tf

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


def load_food_data(batch_size=32, img_size=(224, 224)):
    # Sınıfları karşılaştır ve eksik olanları eşitle
    equalize_classes("data/processed/food/train", "data/processed/food/val")

    # Ortak sınıf adlarını yükle
    train_classes = sorted(os.listdir("data/processed/food/train"))

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/food/train",
        labels="inferred",
        label_mode="categorical",
        class_names=train_classes,  # Ortak sınıf isimlerini kullan
        batch_size=batch_size,
        image_size=img_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/food/val",
        labels="inferred",
        label_mode="categorical",
        class_names=train_classes,  # Ortak sınıf isimlerini doğrulama için de kullan
        batch_size=batch_size,
        image_size=img_size,
    )
    return train_ds, val_ds


def load_calorie_data(batch_size=32, img_size=(224, 224)):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data/processed/calories/train",
        labels="inferred",
        label_mode="int",
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


def compare_classes(train_dir, val_dir):
    """
    Eğitim ve doğrulama setlerindeki sınıfları karşılaştırır.
    """
    train_classes = set(os.listdir(train_dir))
    val_classes = set(os.listdir(val_dir))

    extra_in_train = train_classes - val_classes
    extra_in_val = val_classes - train_classes

    print("Eğitim setinde olup doğrulama setinde olmayanlar:", extra_in_train)
    print("Doğrulama setinde olup eğitim setinde olmayanlar:", extra_in_val)

    return extra_in_train, extra_in_val


def equalize_classes(train_dir, val_dir):
    """
    Eğitim ve doğrulama setlerini aynı sınıflara eşitler.
    Eksik sınıfları boş dizinler olarak ekler.
    """
    extra_in_train, extra_in_val = compare_classes(train_dir, val_dir)

    for cls in extra_in_train:
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for cls in extra_in_val:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)

    print("Eksik sınıflar eşitlendi.")


# Kullanım
if __name__ == "__main__":
    train_dir = "data/processed/food/train"
    val_dir = "data/processed/food/val"

    # Sınıfları eşitle ve kontrol et
    equalize_classes(train_dir, val_dir)

    # Eğitim ve doğrulama setini yükle
    train_ds, val_ds = load_food_data(batch_size=32, img_size=(224, 224))
