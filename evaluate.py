import numpy as np
import cv2
import argparse
import pprint
import os

def read_masks(gt_mask_path, pred_mask_path):
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if gt_mask is None or pred_mask is None:
        raise ValueError("One or both image paths are invalid or images could not be loaded.")
    
    gt_mask = (gt_mask > 128).astype(np.uint8)
    pred_mask = (pred_mask > 128).astype(np.uint8)
    
    return gt_mask, pred_mask

def evaluate_pred(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    true_positive = intersection
    false_positive = pred_mask.sum() - true_positive
    false_negative = gt_mask.sum() - true_positive
    true_negative = np.logical_not(np.logical_or(gt_mask, pred_mask)).sum()
    
    iou = intersection / union if union != 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0
    accuracy = (true_positive + true_negative) / (gt_mask.size) if gt_mask.size != 0 else 0.0
    dice = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) if (2 * true_positive + false_positive + false_negative) != 0 else 0.0
    
    return {
        "IoU": round(iou, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "Accuracy": round(accuracy, 3),
        "Dice Coefficient": round(dice, 3)
    }

def compute_metrics_batch(gt_masks_dir, pred_masks_dir):
    results = {}
    gt_files = set(os.listdir(gt_masks_dir))
    pred_files = set(os.listdir(pred_masks_dir))
    common_files = gt_files.intersection(pred_files)
    
    for filename in common_files:
        gt_mask_path = os.path.join(gt_masks_dir, filename)
        pred_mask_path = os.path.join(pred_masks_dir, filename)
        gt_mask, pred_mask = read_masks(gt_mask_path, pred_mask_path)
        results[filename] = evaluate_pred(gt_mask, pred_mask)
    
    return results

# gt, pred = read_masks("data/masks/done/butterfly.png", "masks_1743012826.658254.png")
# print(evaluate_pred(gt, pred))
# pprint.pprint(compute_metrics_batch("data/masks/done/", "/home/ckapelonis/Downloads/fine-tuned/"))
import pandas as pd

# First run
results1 = compute_metrics_batch("data/masks/done/", "/home/ckapelonis/Downloads/fine-tuned/")

# Second run
results2 = compute_metrics_batch("data/masks/done/", "/home/ckapelonis/Downloads/base_pred/")

# Store all metrics in a dictionary for each model
metrics = ["IoU", "Precision", "Recall", "Accuracy", "Dice Coefficient"]
fine_tuned_metrics = {}
base_metrics = {}

for metric in metrics:
    fine_tuned_metrics[metric] = {file: res[metric] for file, res in results1.items()}
    base_metrics[metric] = {file: res[metric] for file, res in results2.items()}

# Convert dictionaries to DataFrames
df_fine_tuned = pd.DataFrame(fine_tuned_metrics)
df_base = pd.DataFrame(base_metrics)

# Compute mean across all images
mean_fine_tuned = df_fine_tuned.mean().rename("Mean_Fine-Tuned")
mean_base = df_base.mean().rename("Mean_Base")

# Combine means into a single DataFrame
mean_comparison = pd.DataFrame([mean_fine_tuned, mean_base]).T

# Print results
print("\n### Mean Metrics Comparison ###")
print(mean_comparison)
