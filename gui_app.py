import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model_path = 'fruits_veg_model.h5'  # Replace with your actual path
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['Fresh Fruit', 'Rotten Fruit', 'Fresh Vegetable', 'Rotten Vegetable']

# Function to classify an uploaded image
def classify_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
    )
    if file_path:
        try:
            # Load and preprocess the image
            img = load_img(file_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Perform prediction
            predictions = model.predict(img_array)
            predicted_class = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            # Show the result
            messagebox.showinfo("Prediction Result", f"{predicted_class} ({confidence:.2f}% confidence)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to classify image: {e}")

# Function to start real-time camera classification
def classify_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera.")
        return

    messagebox.showinfo("Instructions", "Press 'c' to classify or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1)
        if key == ord('c'):
            img = cv2.resize(frame, (224, 224))
            img = np.expand_dims(img, axis=0) / 255.0

            predictions = model.predict(img)
            label = class_labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Prediction', frame)
            cv2.waitKey(2000)  # Display for 2 seconds
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create the GUI
root = tk.Tk()
root.title("Fruit and Vegetable Quality Checker")
root.geometry("400x300")

# Add buttons for functionality
label = tk.Label(root, text="Fruit & Vegetable Quality Checker", font=("Arial", 16))
label.pack(pady=20)

upload_button = tk.Button(root, text="Upload and Classify Image", command=classify_image, width=30, bg="green", fg="white")
upload_button.pack(pady=10)

camera_button = tk.Button(root, text="Real-Time Camera Classification", command=classify_camera, width=30, bg="blue", fg="white")
camera_button.pack(pady=10)

exit_button = tk.Button(root, text="Exit", command=root.quit, width=30, bg="red", fg="white")
exit_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
