import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and Preprocess the Data
def load_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    
    # Multiclass categories (update folder names to match these)
    categories = ["no_fracture", "minor", "moderate", "severe", "critical"]
    
    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist. Skipping...")
            continue
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Load dataset
train_dir = "dataset/train"
val_dir = "dataset/val"

X_train, y_train = load_data(train_dir)
X_val, y_val = load_data(val_dir)

# Normalize
X_train = X_train / 255.0
X_val = X_val / 255.0

# Reshape for input
X_train = X_train.reshape(-1, 128, 128, 1)
X_val = X_val.reshape(-1, 128, 128, 1)

# One-hot encode labels
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=5)
y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=5)

# Define U-Net encoder
def unet_encoder(inputs):
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    return p4

# Build Model
inputs = keras.Input(shape=(128, 128, 1))
encoded_features = unet_encoder(inputs)

flattened = layers.Flatten()(encoded_features)
dense1 = layers.Dense(128, activation='relu')(flattened)
dropout = layers.Dropout(0.5)(dense1)
output = layers.Dense(5, activation='softmax')(dropout)

model = keras.Model(inputs=inputs, outputs=output)

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train Model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=25,
    batch_size=16
)

# Evaluate Model
val_preds = model.predict(X_val)
y_pred = np.argmax(val_preds, axis=1)
y_true = np.argmax(y_val_cat, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=[
    "No Fracture", "Minor", "Moderate", "Severe", "Critical"
]))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No", "Minor", "Moderate", "Severe", "Critical"],
            yticklabels=["No", "Minor", "Moderate", "Severe", "Critical"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save Model
model.save("fracture_detection_multiclass_unet.keras", save_format="keras")

# Prediction on New Image
def predict_fracture_severity(model, img_path):
    class_names = [
        "No Fracture (0%)",
        "Minor (0–25%)",
        "Moderate (26–50%)",
        "Severe (51–75%)",
        "Critical (76–100%)"
    ]
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2f})")
