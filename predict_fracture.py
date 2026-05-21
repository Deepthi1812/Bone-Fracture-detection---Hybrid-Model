# source venv/bin/activate
# python predict_fracture.py
import tensorflow as tf 
import cv2
import numpy as np
import os


model = tf.keras.models.load_model("fracture_detection_model.keras")

# Function to preprocess a new image
def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    img = cv2.resize(img, img_size) 
    img = img / 255.0  # Normalize
    img = img.reshape(1, 224, 224, 1)  
    return img

def predict_image(image_path):
    
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    probability = prediction[0][0] 
    if probability > 0.5:
        print(f"Image: {image_path} -> Predicted: Non-Fractured ({probability:.2f})")
    else:
        print(f"Image: {image_path} -> Predicted: Fractured ({probability:.2f})")

#  test images
test_images_dir = "test_images"  

test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith((".jpg", ".png", ".jpeg"))]

for img_path in test_images:
    predict_image(img_path)
