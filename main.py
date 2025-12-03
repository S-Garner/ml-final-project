#!/usr/bin/env python3

from datasets                   import load_dataset
from sklearn.model_selection    import train_test_split
from PIL                        import Image
from tensorflow.keras           import layers, models
import numpy as np
import tensorflow as tf


# ========================================
# LOAD DATASET
# ========================================

def load_in_dataset():
    return load_dataset('microsoft/cats_vs_dogs')


# ========================================
# IMAGE PROCESSING
# ========================================

def process_image(image, size):
    image = image.convert("RGB")
    image = image.resize((size, size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def build_xy(dataset, size):
    images, labels = [], []
    for item in dataset['train']:
        arr = process_image(item["image"], size)
        images.append(arr)
        labels.append(item["labels"])

    x = np.stack(images)
    y = np.array(labels)
    return x, y


def split_data(x, y):
    return train_test_split(
        x, y,
        test_size=0.2,
        stratify=y,
        random_state=0
    )


# ========================================
# CNN MODEL
# ========================================

def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ========================================
# TRANSFER LEARNING MODEL (Returns model AND base_model)
# ========================================

def build_transfer_model(input_shape):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model, base_model


# ========================================
# PREDICTION FUNCTION
# ========================================

def predict_image(model, path, size):
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0][0]
    label = "DOG" if pred > 0.5 else "CAT"
    print(f"{path}: {label} ({pred:.4f})")


# ========================================
# METRICS PRINTING
# ========================================

def print_metrics(history, label):
    print(f"\n=== FINAL METRICS FOR {label} ===")

    train_acc = history.history["accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_acc = history.history["val_accuracy"][-1]
    val_loss = history.history["val_loss"][-1]

    print(f"Train Accuracy:      {train_acc:.4f}")
    print(f"Train Loss:          {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Loss:     {val_loss:.4f}")
    print("")


# ========================================
# MAIN
# ========================================

def main():
    print("Loading dataset...\n")
    dataset = load_in_dataset()

    IMAGE_SIZE = 128

    print("Processing images...\n")
    x, y = build_xy(dataset, IMAGE_SIZE)

    print("Splitting data...\n")
    x_train, x_test, y_train, y_test = split_data(x, y)

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    print("Building models...\n")
    cnn = build_cnn_model(input_shape)
    transfer, base_model = build_transfer_model(input_shape)

    print("Training CNN (baseline)...")
    cnn_history = cnn.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=32
    )

    print("Training transfer model (frozen base)...")
    transfer_history = transfer.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=32
    )

    # ========================================
    # FINE-TUNING
    # ========================================

    print("\nStarting fine-tuning...")

    base_model.trainable = True

    # Freeze all but the last 20 layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    transfer.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_history = transfer.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=3,
        batch_size=32
    )

    # ========================================
    # OUTPUT METRICS
    # ========================================

    print_metrics(cnn_history,             "CNN Model")
    print_metrics(transfer_history,        "Transfer Model (Frozen)")
    print_metrics(fine_tune_history,       "Transfer Model (Fine-Tuned)")

    # ========================================
    # TEST ON A CUSTOM IMAGE
    # ========================================

    print("\nTesting on custom image:\n")
    predict_image(transfer, "data/images/dog_01.jpg", IMAGE_SIZE)


if __name__ == "__main__":
    main()
