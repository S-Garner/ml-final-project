#!/usr/bin/env python3

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ========================================
# kNN MODEL
# ========================================

def train_knn(x_train, y_train, k):
    n = x_train.shape[0]
    flat = x_train.reshape(n, -1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(flat, y_train)
    return knn


def knn_accuracy(knn, x_test, y_test):
    n = x_test.shape[0]
    flat = x_test.reshape(n, -1)
    preds = knn.predict(flat)
    return np.mean(preds == y_test)


# ========================================
# METRICS
# ========================================

def print_metrics(history, label):
    print(f"\n=== FINAL METRICS FOR {label} ===")
    final_train_acc = history.history["accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    final_val_loss = history.history["val_loss"][-1]

    print(f"Train Accuracy:      {final_train_acc:.4f}")
    print(f"Train Loss:          {final_train_loss:.4f}")
    print(f"Validation Accuracy: {final_val_acc:.4f}")
    print(f"Validation Loss:     {final_val_loss:.4f}\n")

def predict_image(model, path, size):
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size))

    arr = np.asarray(image, dtype=np.float32) / 255.0

    # CNN expects shape (1, H, W, C)
    if isinstance(model, tf.keras.Model):
        arr_expanded = np.expand_dims(arr, axis=0)
        pred = model.predict(arr_expanded)[0][0]
        label = "DOG" if pred > 0.5 else "CAT"
        print(f"{path}: CNN -> {label} ({pred:.4f})")
        return

    # kNN expects shape (1, H*W*C)
    flat = arr.reshape(1, -1)
    pred = model.predict(flat)[0]

    label = "DOG" if pred == 1 else "CAT"
    print(f"{path}: kNN -> {label}")

def plot_cnn_history(history, model_id="cnn_model"):
    # ----- Accuracy -----
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{model_id} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig(f"{model_id}_accuracy.png")
    plt.close()

    # ----- Loss -----
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(f"{model_id} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.savefig(f"{model_id}_loss.png")
    plt.close()


def plot_knn_results(k_values, accuracies, model_id="knn"):
    plt.figure()
    plt.bar([str(k) for k in k_values], accuracies)
    plt.title(f"{model_id} - Accuracy by k Value")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.savefig(f"{model_id}_accuracy.png")
    plt.close()

# ========================================
# MAIN
# ========================================

def main():
    print("Loading dataset...\n")
    dataset = load_in_dataset()

    IMAGE_SIZE = 64

    print("Processing images...\n")
    x, y = build_xy(dataset, IMAGE_SIZE)

    print("Splitting data...\n")
    x_train, x_test, y_train, y_test = split_data(x, y)

    print("Shapes:")
    print(f"x_train: {x_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    # ---- CNN ----
    cnn = build_cnn_model(input_shape)
    cnn_history = cnn.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=32,
        shuffle=True
    )

    # ---- kNN ----
    print("\nTraining kNN (k=5)...")
    knn = train_knn(x_train, y_train, k=5)
    knn_acc = knn_accuracy(knn, x_test, y_test)
    print(f"kNN Accuracy: {knn_acc:.4f}")

    # ---- kNN ----
    print("\nTraining kNN models...")

    k_values = [1, 3, 5, 7, 11]
    accuracies = []

    for k in k_values:
        print(f"Training kNN (k={k})...")
        knn = train_knn(x_train, y_train, k)
        acc = knn_accuracy(knn, x_test, y_test)
        accuracies.append(acc)
        print(f"Accuracy for k={k}: {acc:.4f}")

    # Create plot
    plot_knn_results(k_values, accuracies, model_id="knn_model")

    plot_cnn_history(cnn_history, model_id="cnn_model")

    # ---- Output Metrics ----
    print_metrics(cnn_history, label="CNN Model")

    predict_image(cnn, "data/images/dog_01.jpg", IMAGE_SIZE)
    predict_image(knn, "data/images/dog_01.jpg", IMAGE_SIZE)

if __name__ == "__main__":
    main()
