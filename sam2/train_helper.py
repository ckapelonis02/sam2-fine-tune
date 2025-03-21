import hydra
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def read_dataset(images_path, masks_path, file_names):
    data = [
        {
        "image": os.path.join(images_path, str(file)) + ".jpg",
        "masks": os.path.join(masks_path, str(file)) + ".png",
        }
        for file in file_names
    ]

    return data

def read_batch(data_dict, index, max_res=1024):
    ent = data_dict[index]
    img = cv2.imread(ent["image"])[..., ::-1]  
    ann_map = cv2.imread(ent["masks"], cv2.IMREAD_GRAYSCALE)  

    r = np.min([max_res / img.shape[1], max_res / img.shape[0]])  
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))  
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)  

    # Ensure binary
    ann_map = (ann_map > 127).astype(np.uint8) * 255  

    masks = []
    points = []

    # Find contours
    contours, _ = cv2.findContours((255 - ann_map).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if len(contour) >= 3:
            mask = np.zeros_like(ann_map)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Compute centroid using image moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if (cx != 0 and cy != 0):
                    points.append([[cx, cy]])
                    masks.append(mask)

    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def visualize_entry(img, masks, points):
    # Plot the input image
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title("Input Image (Resized)")
    plt.axis("on")
    plt.show()

    # Plot the combined binary annotation mask
    plt.figure(figsize=(8, 8))
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)  
    plt.imshow(combined_mask, cmap='gray')  
    plt.title("Combined Mask (Tesserae in White)")
    plt.axis("on")
    plt.show()

    # Plot the image with truly random points inside tesserae
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    for point in points:
        plt.plot(point[0][0], point[0][1], 'ro', markersize=2)
    plt.title("Image with Randomly Distributed Points")
    plt.axis("on")
    plt.show()

def visualize_training_results(image, masks, predicted_mask, iou, itr):
    # Plot the input image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title("Input Image (Resized)")
    plt.axis("on")
    plt.show()

    # Plot the ground truth mask
    plt.figure(figsize=(8, 8))
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)
    plt.imshow(combined_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis("on")
    plt.show()

    # Plot the predicted mask
    plt.figure(figsize=(8, 8))
    plt.imshow(predicted_mask[0, 0].cpu().detach().numpy(), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("on")
    plt.show()

    # Plot the IoU score over time
    plt.figure(figsize=(8, 8))
    plt.plot(itr, iou, label="IoU")
    plt.xlabel('Iterations')
    plt.ylabel('IoU')
    plt.title("IoU Over Time")
    plt.legend()
    plt.show()
