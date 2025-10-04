import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

class LeafSegmenter:
    def __init__(self):
        self.kernel_size = (5, 5)
        self.sigma = 1.0
        self.masks = None
        self.target_size = (256, 256)  # Match model's expected input size
        
    def segment(self, img):
        """
        Segment the leaf from the background using color-based segmentation
        Args:
            img: numpy array of the image
        Returns:
            mask: binary mask of the segmented leaf
        """
        if img is None or img.size == 0:
            return None
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define range for green color
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Create mask for green color
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones(self.kernel_size, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to smooth edges
        mask = cv2.GaussianBlur(mask, self.kernel_size, self.sigma)
        
        # Check if mask contains any leaf area
        if np.sum(mask) == 0:
            return None
            
        return mask
        
    def segment_leaf(self, image_path):
        """
        Segment the leaf from the background using color-based segmentation
        Args:
            image_path: path to the image file
        Returns:
            segmented: segmented image with background removed
        """
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Get the mask
        mask = self.segment(img)
        if mask is None:
            raise ValueError("No leaf area detected in the image")
        
        # Apply the mask to the original image
        segmented = cv2.bitwise_and(img, img, mask=mask)
        
        return segmented
        
    def extract_leaf(self, img, mask):
        """
        Extract the leaf from the image using the provided mask
        Args:
            img: numpy array of the image
            mask: binary mask of the leaf
        Returns:
            leaf_img: image with only the leaf visible
        """
        if mask is None or mask.size == 0:
            raise ValueError("Invalid mask provided")
            
        # Ensure mask is binary
        if np.max(mask) > 1:
            mask = mask / 255.0
            
        # Convert mask to 3 channels
        mask_3d = np.stack([mask] * 3, axis=-1)
        
        # Extract leaf
        leaf_img = img * mask_3d
        
        return leaf_img

def preprocess_image(img_path, target_size=(256, 256)):  # Updated default size
    """
    Preprocess image for model prediction
    """
    # First segment the leaf
    segmenter = LeafSegmenter()
    try:
        segmented_img = segmenter.segment_leaf(img_path)
        # Save the segmented image temporarily
        temp_path = img_path.replace('.jpg', '_segmented.jpg')
        cv2.imwrite(temp_path, segmented_img)
        
        # Load and preprocess the segmented image
        img = image.load_img(temp_path, target_size=target_size)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return preprocess_input(arr)
    except Exception as e:
        # If segmentation fails, fall back to original image
        img = image.load_img(img_path, target_size=target_size)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        return preprocess_input(arr)
