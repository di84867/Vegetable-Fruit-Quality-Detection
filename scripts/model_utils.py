from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import os
def load_model():
    """Defines, loads, and returns the pre-trained model."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(4, activation='softmax')  # 4 classes
    ])

    # Construct an absolute path to the model file to avoid pathing issues.
    # This assumes 'fruits_veg_model.h5' is in the project's root directory.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'fruits_veg_model.h5')

    # Load the weights from the absolute path
    model.load_weights(model_path)
    return model

CLASS_LABELS = [ 'Fresh Fruit', 'Fresh Vegetable', 'Rotten Fruit', 'Rotten Vegetable' ]