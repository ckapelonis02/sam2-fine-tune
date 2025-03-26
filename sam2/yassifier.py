import cv2
import numpy as np

def remove_thin_connections(mask, min_gap=2):
    """
    Removes thin pixel connections between tesserae by applying morphological operations.
    :param mask: Binary mask (numpy array) where tesserae are white (255) and background is black (0)
    :param min_gap: Minimum thickness of connections to be removed.
    :return: Processed binary mask.
    """
    # Convert to binary if not already
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Create an empty mask
    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for i in range(1, num_labels):  # Skip background label 0
        component = (labels == i).astype(np.uint8) * 255
        
        # Check if eroding and dilating removes small connections
        eroded = cv2.erode(component, np.ones((min_gap, min_gap), np.uint8), iterations=1)
        dilated = cv2.dilate(eroded, np.ones((min_gap, min_gap), np.uint8), iterations=1)
        
        # Add cleaned component back
        cleaned_mask = cv2.bitwise_or(cleaned_mask, dilated)
    
    return cleaned_mask

# Example usage:
if __name__ == "__main__":
    mask = cv2.imread("blended_mask.png", cv2.IMREAD_GRAYSCALE)
    cleaned_mask = remove_thin_connections(mask, min_gap=3)
    cv2.imwrite("blended_mask_2.png", cleaned_mask)
