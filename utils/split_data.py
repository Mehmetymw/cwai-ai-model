import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Veri setini eğitim ve doğrulama olarak böler.
    :param source_dir: Orijinal veri setinin olduğu klasör.
    :param train_dir: Eğitim setinin kaydedileceği klasör.
    :param val_dir: Doğrulama setinin kaydedileceği klasör.
    :param split_ratio: Eğitim seti oranı (örn., %80 için 0.8).
    """
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        # Her sınıfın görüntülerini alıyoruz
        class_dir = os.path.join(source_dir, cls)
        images = os.listdir(class_dir)
        random.shuffle(images)
        
        # Sınıf için eğitim ve doğrulama dizinleri oluşturuyoruz
        train_class_dir = os.path.join(train_dir, cls)
        val_class_dir = os.path.join(val_dir, cls)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Görüntüleri bölüyoruz
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Görüntüleri ilgili dizinlere taşıyoruz
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_class_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_class_dir, img))

    print("Veri seti başarıyla bölündü ve taşındı.")

# Ana dizinler
source_dir = "data/raw/food-101/images"
train_dir = "data/processed/food/train"
val_dir = "data/processed/food/val"

# Betiği çalıştır
if __name__ == "__main__":
    split_data(source_dir, train_dir, val_dir)
