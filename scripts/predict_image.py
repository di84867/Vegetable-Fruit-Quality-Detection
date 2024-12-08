import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('fruits_veg_model.h5')

# Class labels
class_labels = ['Fresh Fruit', 'Rotten Fruit', 'Fresh Vegetable', 'Rotten Vegetable']

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
