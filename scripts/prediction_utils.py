import numpy as np
from .model_utils import CLASS_LABELS
def get_prediction_details(model, image_array, confidence_threshold, rejection_threshold):
    """
    Performs prediction and validates it against confidence and rejection thresholds.

    Args:
        model: The trained Keras model.
        image_array: The preprocessed image array for prediction.
        confidence_threshold (float): The minimum confidence for the top prediction.
        rejection_threshold (float): The maximum allowed confidence for any other class.

    Returns:
        A tuple (is_valid, details) where `is_valid` is a boolean and `details`
        is a dictionary containing prediction info if valid.
    """
    predictions = model.predict(image_array, verbose=0)
    confidences = predictions[0] * 100

    top_confidence = np.max(confidences)
    predicted_class_index = np.argmax(confidences)

    # Create a mask to check all other classes
    mask = np.ones(len(confidences), dtype=bool)
    mask[predicted_class_index] = False
    other_confidences = confidences[mask]

    is_valid = top_confidence >= confidence_threshold and np.all(other_confidences < rejection_threshold)

    if is_valid:
        details = {'label': CLASS_LABELS[predicted_class_index], 'confidence': top_confidence}
        return True, details
    return False, None