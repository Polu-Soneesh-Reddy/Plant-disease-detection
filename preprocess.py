import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Preprocess an image for model prediction.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for model prediction
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to model's expected input size (512x512)
        img = cv2.resize(img, (512, 512))
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize pixel values
        
        logger.info(f"Successfully preprocessed image: {image_path}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        raise 