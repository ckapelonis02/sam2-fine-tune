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
from sam2.test_helper import test_generator
from sam2.train_helper import cleanup

cleanup()

# Configurations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module('sam2', version_base='1.2')

sam2_model = build_sam2(
    config_file="../sam2_configs/sam2_hiera_t.yaml",
    ckpt_path="/kaggle/input/segment-anything-2/pytorch/sam2-hiera-tiny/1/sam2_hiera_tiny.pt",
    device="cuda",
    apply_postprocessing=False
)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=128,
    points_per_batch=16,
    pred_iou_thresh=0.8,
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
    load_model="model6500.torch"
)

start_time = time.time()
test_generator(
    mask_generator=mask_generator,
    img_path="/kaggle/input/mosaic-images/butterfly.jpg",
    output_path=f"results/masks_{time.time()}.png",
    rows=1,
    cols=1,
    max_mask_crop_region=0.2,
    show_masks=True
)
print(f"Time taken: {time.time() - start_time}")
