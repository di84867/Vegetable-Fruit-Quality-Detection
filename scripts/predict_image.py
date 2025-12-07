import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')  # 4 classes
])

# Load the weights
model.load_weights('fruits_veg_model.h5')

# Class labels
class_labels = ['Fresh Fruit', 'Fresh Vegetable', 'Rotten Fruit', 'Rotten Vegetable']

# Function to classify an uploaded image
def classify_image(image_path):
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        print(f"Predicted: {predicted_class} ({confidence:.2f}% confidence)")
    except Exception as e:
        print(f"Error: {e}")

# User uploads an image
upload_path = 'uploads/uploaded_image.jpg'
print("Place your image in the 'uploads/' folder and name it 'uploaded_image.jpg'")
input("Press Enter once the image is ready...")

# Classify the uploaded image
if os.path.exists(upload_path):
    classify_image(upload_path)
else:
    print("No image found in the 'uploads/' folder.")
