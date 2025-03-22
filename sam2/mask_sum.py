import os
import numpy as np
import cv2

# Set the folder path containing the masks
folder_path = "project-1-at-2025-03-22-15-27-b195d3d9"  # Change this to your actual path

# Get all PNG images in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

# Initialize an accumulator for the blended mask
blended_mask = None

for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    
    if blended_mask is None:
        blended_mask = np.zeros_like(img, dtype=np.uint16)  # Use uint16 to avoid overflow
    
    blended_mask += img  # Add mask values

# Clip values to [0, 255] and convert back to uint8
blended_mask = np.clip(blended_mask, 0, 255).astype(np.uint8)

# Save the final blended mask
cv2.imwrite("blended_mask.png", blended_mask)

print("Blended mask saved as blended_mask.png")
