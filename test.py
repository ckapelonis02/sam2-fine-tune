import numpy as np
import torch
import cv2
import hydra
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.inference_helper import show_anns

start_time = time.time()

# Configurations
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module('sam2', version_base='1.2')
sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "../sam2_configs/sam2_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda", apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=16,
    points_per_batch=4,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.8,
    stability_score_offset=1.0,
    mask_threshold=0.6,
    box_nms_thresh=0.1,
    crop_n_layers=2,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=0.5,
    crop_n_points_downscale_factor=2,
    point_grids=None,
    min_mask_region_area=25.0,
    # output_mode="binary_mask",
    # use_m2m=True,
    # multimask_output=False,
    # load_model="checkpoints/model.torch"
)


image = Image.open('data/butterfly.jpg')
width, height = image.size

# Define crop regions
crop_regions = []
rows, cols = 4, 4
cell_width, cell_height = width // cols, height // rows

for i in range(rows):
    for j in range(cols):
        crop_regions.append((j * cell_width, i * cell_height, (j + 1) * cell_width, (i + 1) * cell_height))

# Create an empty canvas for the final stitched image
final_mask = np.zeros((height, width, 3), dtype=np.uint8)

# Process each quadrant
for i, crop in enumerate(crop_regions):
    print(f"Processing {i+1}/{len(crop_regions)}")
    cropped_image = image.crop(crop)
    cropped_image_np = np.array(cropped_image.convert("RGB"))
    masks = mask_generator.generate(cropped_image_np)
    
    print(f"{len(masks)} masks found")
    
    plt.figure(figsize=(12, 12))
    plt.imshow(cropped_image_np)
    # show_anns(masks)
    # plt.axis('off')
    # plt.show()
    
    # Generate mask for this quadrant
    mask_overlay = np.zeros_like(cropped_image_np)

    # Define a max area threshold (e.g., 50% of crop area)
    max_area_threshold = (cell_width * cell_height) * 0.5  # 50% of the cropped region

    for mask in masks:
        mask_area = np.sum(mask['segmentation'])  # Count mask pixels

        if mask_area < max_area_threshold:  # Only keep masks below the threshold
            mask_overlay[mask['segmentation']] = (255, 255, 255)  # White for visibility
    
    # Place the mask in the final image
    x1, y1, x2, y2 = crop
    final_mask[y1:y2, x1:x2] = mask_overlay

# Save the final stitched mask
os.makedirs("results", exist_ok=True)
final_mask_pil = Image.fromarray(final_mask)
final_mask_pil.save("results/stitched_segmentation.png")
print("Final stitched segmentation saved at results/stitched_segmentation.png")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
