import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from scripts.predict_image import classify_image_from_path
from scripts.predict_camera import run_camera_classification

print("Welcome to the Fruit and Vegetable Quality Checker!")
print("1. Upload and Classify Image")
print("2. Real-Time Camera Classification")
choice = input("Enter your choice (1/2): ")

if choice == '1':
    # Logic from predict_image.py is now integrated here
    upload_path = 'uploads/uploaded_image.jpg'
    print(f"\nPlease place your image at: {os.path.join(project_root, upload_path)}")
    input("Press Enter once the image is ready...")

    if os.path.exists(upload_path):
        pred_class, conf = classify_image_from_path(upload_path)
        print(f"\nPredicted: {pred_class} ({conf:.2f}% confidence)")
    else:
        print(f"\nError: Image not found at {upload_path}")

elif choice == '2':
    # Logic from predict_camera.py is now called directly
    run_camera_classification()
else:
    print("Invalid choice. Exiting.")
