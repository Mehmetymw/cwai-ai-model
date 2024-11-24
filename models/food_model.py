import tensorflow as tf
from tensorflow.keras import layers, models

def create_food_classification_model(input_shape=(224, 224, 3), num_classes=101):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # ResNet50 tabanlı önceden eğitilmiş model
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = True  # Modelin tüm katmanlarını eğitilebilir yap
    
    # İlk katmanları dondurma (freeze)
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    # Tamamen bağlı katmanlar (Classifier)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    # Model derleme
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    from utils.data_loader import load_food_data
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    # Eğitim ve doğrulama verilerini yükleme
    train_data, val_data, num_classes = load_food_data(batch_size=32, img_size=(224, 224))

    # Model oluşturma
    model = create_food_classification_model(input_shape=(224, 224, 3), num_classes=num_classes)

    # Callbacks tanımlama
    checkpoint = ModelCheckpoint(
        filepath="models/weights/food_weights.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1
    )

    # Modeli eğitme
    history = model.fit(
        train_data,
        epochs=1,
        validation_data=val_data,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )

    # Model ağırlıklarını kaydetme
    model.save("models/weights/food_weights.keras")
    print("Gıda tanıma modeli başarıyla eğitildi ve kaydedildi.")
