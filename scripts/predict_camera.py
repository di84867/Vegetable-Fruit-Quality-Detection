import cv2
import numpy as np
import tensorflow as tf

# Reconstruct the model architecture
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

# Start video capturemahabharat ep 140
cap = cv2.VideoCapture(0)

print("Press 'c' to capture and classify an image, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live video feed
    cv2.imshow('Camera Feed', frame)

    # Wait for user input
    key = cv2.waitKey(1)
    if key == ord('c'):  # Capture and classify
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0

        # Perform prediction
        predictions = model.predict(img)
        label = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        print(f"Predicted: {label} ({confidence:.2f}% confidence)")
        cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Prediction', frame)
        cv2.waitKey(2000)  # Display for 2 seconds
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
