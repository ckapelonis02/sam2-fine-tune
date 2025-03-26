import numpy as np
import torch
import tqdm
import cv2
import hydra
import matplotlib.pyplot as plt
import os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=2) 

    ax.imshow(img)

def test_generator(mask_generator, img_path, output_path, rows, cols, max_mask_crop_region, overlap_ratio=0.1, show_masks=False):
    image = Image.open(img_path)
    width, height = image.size

    crop_regions = []
    cell_width, cell_height = width // cols, height // rows

    # Calculate overlap amount
    overlap_width = int(cell_width * overlap_ratio)
    overlap_height = int(cell_height * overlap_ratio)

    for i in range(rows):
        for j in range(cols):
            # Define crop regions with overlap
            left = max(j * cell_width - overlap_width, 0)
            upper = max(i * cell_height - overlap_height, 0)
            right = min((j + 1) * cell_width + overlap_width, width)
            lower = min((i + 1) * cell_height + overlap_height, height)
            crop_regions.append((left, upper, right, lower))

    final_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for i, crop in enumerate(crop_regions):
        print(f"Processing {i+1} of {len(crop_regions)}")
        cropped_image = image.crop(crop)
        cropped_image_np = np.array(cropped_image.convert("RGB"))
        masks = mask_generator.generate(cropped_image_np)

        print(f"{len(masks)} masks found")

        if show_masks:     
            plt.figure(figsize=(12, 12))
            plt.imshow(cropped_image_np)
            show_anns(masks)
            plt.axis('off')
            plt.show()

        mask_overlay = np.zeros_like(cropped_image_np, dtype=np.uint8)
        max_area_threshold = (cell_width * cell_height) * max_mask_crop_region

        for mask in masks:
            mask_area = np.sum(mask['segmentation'])  # Count mask pixels
            if mask_area < max_area_threshold:  # Only keep masks below the threshold
                mask_overlay[mask['segmentation']] = (255, 255, 255)  # White for visibility

        # Apply logical OR instead of overwriting
        x1, y1, x2, y2 = crop
        final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], mask_overlay)

    final_mask_pil = Image.fromarray(final_mask)
    final_mask_pil.save(output_path)
    print(f"Final stitched segmentation saved as {output_path}")
