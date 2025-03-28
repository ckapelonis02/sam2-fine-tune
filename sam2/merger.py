import os
import numpy as np
import cv2

# Set the folder paths
folder_path_1 = "fine-tuned-hd"  # Change to actual folder path
folder_path_2 = "base_pred"  # Change to actual folder path
output_folder = "output_folder"  # Change to desired output folder

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all PNG images in the first folder
image_files_1 = {f for f in os.listdir(folder_path_1) if f.endswith(".png")}
image_files_2 = {f for f in os.listdir(folder_path_2) if f.endswith(".png")}

# Find common image names
common_files = image_files_1.intersection(image_files_2)

for img_file in common_files:
    img_path_1 = os.path.join(folder_path_1, img_file)
    img_path_2 = os.path.join(folder_path_2, img_file)
    
    # Read images in grayscale
    img1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path_2, cv2.IMREAD_GRAYSCALE)
    
    # Ensure both images have the same shape
    if img1.shape != img2.shape:
        print(f"Skipping {img_file}: shape mismatch")
        continue
    
    # Use uint16 to avoid overflow, then add the images
    blended_mask = np.clip(img1.astype(np.uint16) + img2.astype(np.uint16), 0, 255).astype(np.uint8)
    
    # Save the blended mask in the output folder
    output_path = os.path.join(output_folder, img_file)
    cv2.imwrite(output_path, blended_mask)
    
print(f"Blended masks saved in {output_folder}")
