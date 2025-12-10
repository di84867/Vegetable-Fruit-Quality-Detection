import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from .model_utils import load_model
from .model_utils import CLASS_LABELS

def classify_image_from_path(image_path):
    """Loads an image, preprocesses it, and returns the prediction and confidence."""
    model = load_model()
    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions[0])]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence
    
if __name__ == '__main__':
    upload_path = 'uploads/uploaded_image.jpg'
    print("Place your image in the 'uploads/' folder and name it 'uploaded_image.jpg'")
    input("Press Enter once the image is ready...")

    if os.path.exists(upload_path):
        pred_class, conf = classify_image_from_path(upload_path)
        print(f"Predicted: {pred_class} ({conf:.2f}% confidence)")
    else:
        print("No image found in the 'uploads/' folder.")
