import cv2
import numpy as np
from .model_utils import load_model, CLASS_LABELS

def run_camera_classification():
    """Initializes camera and runs real-time classification loop."""
    model = load_model()
    cap = cv2.VideoCapture(0)
    print("\nPress 'c' to capture and classify an image, or 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            img = cv2.resize(frame, (224, 224))
            img_array = np.expand_dims(img, axis=0) / 255.0
            predictions = model.predict(img_array)
            label = CLASS_LABELS[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            print(f"-> Predicted: {label} ({confidence:.2f}% confidence)")
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
