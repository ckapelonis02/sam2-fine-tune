import optuna
import numpy as np
import time
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.test_helper import test_generator
from evaluate import *

def objective(trial):
    points_per_side = 128
    points_per_batch = 32
    pred_iou_thresh = trial.suggest_float('pred_iou_thresh', 0.5, 0.9)
    stability_score_thresh = trial.suggest_float('stability_score_thresh', 0.7, 0.95)
    stability_score_offset = trial.suggest_float('stability_score_offset', 0.7, 1.2)
    mask_threshold = trial.suggest_float('mask_threshold', 0.0, 0.6)
    box_nms_thresh = 0.7
    crop_n_layers = 2
    crop_nms_thresh = 0.7
    crop_overlap_ratio = 0.3
    crop_n_points_downscale_factor = 2
    min_mask_region_area = 25.0
    use_m2m = False

    sam2_model = build_sam2(
        config_file="../sam2_configs/sam2_hiera_t.yaml",
        ckpt_path="/kaggle/input/segment-anything-2/pytorch/sam2-hiera-tiny/1/sam2_hiera_tiny.pt",
        device="cuda",
        apply_postprocessing=False
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        mask_threshold=mask_threshold,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        use_m2m=use_m2m
    )

    img_path = "/kaggle/input/evaluation-dataset/images_set/butterfly.jpg"
    output_path = "/kaggle/working/sam2-fine-tune/results/butterfly.png"

    start_time = time.time()
    test_generator(
        mask_generator=mask_generator,
        img_path=img_path,
        output_path=output_path,
        rows=1,
        cols=1,
        max_mask_crop_region=0.1,
        show_masks=False
    )
    print(f"Test run took {time.time() - start_time} seconds")

    gt, pred = read_masks("/kaggle/input/evaluation-dataset/masks_set/butterfly.png", output_path)
    metrics = evaluate_pred(gt, pred)
    iou_score = metrics['IoU']

    return iou_score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best Hyperparameters:", study.best_params)
print("Best IoU Score:", study.best_value)
