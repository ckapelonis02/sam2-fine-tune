import os
import cv2
import numpy as np

def closest_factors(n):
    """Find two factors of n that are as close as possible (to form a nearly square grid)."""
    factors = [(i, n // i) for i in range(1, int(np.sqrt(n)) + 1) if n % i == 0]
    return min(factors, key=lambda x: abs(x[0] - x[1]))  # Choose the closest pair

def split_image(img, divisions=24):
    h, w, _ = img.shape
    
    # Find best grid size (rows, cols) to split the image
    rows, cols = closest_factors(divisions)
    
    sub_h, sub_w = h // rows, w // cols  # Compute approximate sub-image dimensions
    sub_images = []

    for i in range(rows):
        for j in range(cols):
            y_start, y_end = i * sub_h, (i + 1) * sub_h if i < rows - 1 else h
            x_start, x_end = j * sub_w, (j + 1) * sub_w if j < cols - 1 else w
            sub_images.append(img[y_start:y_end, x_start:x_end])

    return sub_images

def process_images_in_folder(input_folder, divisions=24):
    # Get all .jpg files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    # image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    count = 1  # Counter for naming sub-images sequentially

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)  # Read image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (if needed)

        sub_images = split_image(img, divisions)  # Split the image into sub-images

        # Save each sub-image
        for sub_img in sub_images:
            filename = f"{count}.jpg"  # Name sub-image sequentially
            # filename = f"{count}.png"  # Name sub-image sequentially
            sub_img_bgr = cv2.cvtColor(sub_img, cv2.COLOR_RGB2BGR)  # Convert back to BGR
            cv2.imwrite(filename, sub_img_bgr)  # Save the sub-image
            print(f"Saved: {filename}")
            count += 1  # Increment the counter for the next sub-image

# Example usage:
input_folder = "/home/ckapelonis/Desktop/thesis/thesis-code/mosaic_generator/data/output_images_original"  # Specify the folder containing your .jpg files
# input_folder = "/home/ckapelonis/Desktop/thesis/thesis-code/mosaic_generator/data/output_images_mask"  # Specify the folder containing your .jpg files
process_images_in_folder(input_folder, divisions=20)  # Process all images and split them
