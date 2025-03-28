import torch
import numpy as np
import random
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.train_helper import *

cleanup()

# Model Initialization
sam2_model = build_sam2(
    config_file="../sam2_configs/sam2_hiera_t.yaml",
    ckpt_path="/kaggle/input/segment-anything-2/pytorch/sam2-hiera-tiny/1/sam2_hiera_tiny.pt",
    device="cuda",
    apply_postprocessing=False
)
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)

# Optimizer & Scheduler
optimizer = optim.AdamW(predictor.model.parameters(), lr=1e-5, weight_decay=4e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000, eta_min=1e-7)
scaler = torch.cuda.amp.GradScaler()

# Dataset Configuration
data_size = 2000
file_names = list(range(1, data_size + 1))
train_size = int(0.8 * data_size)
train_files, val_files = file_names[:train_size], file_names[train_size:]

train_data = read_dataset("/kaggle/input/data-2k-cropped/images", "/kaggle/input/data-2k-cropped/masks", train_files)
val_data = read_dataset("/kaggle/input/data-2k-cropped/images", "/kaggle/input/data-2k-cropped/masks", val_files)

# Training Parameters
max_masks = 150
epochs = 10
best_val_iou = 0.0
gradient_accumulation_steps = 4
patience = 3  # Number of epochs to wait before early stopping
no_improvement_count = 0  # Counter for no improvement in validation IoU

# Training Loop
for epoch in range(epochs):
    total_iou = 0
    total_loss = 0
    random.shuffle(train_files)
    
    print(f"\nEpoch {epoch+1}/{epochs}")

    for itr in tqdm(range(train_size), desc="Training Progress"):
        with torch.cuda.amp.autocast():
            image, masks, input_point, input_label = read_batch(train_data, itr % train_size, max_masks)
            prd_mask, prd_scores, gt_mask = process_batch(predictor, image, masks, input_point, input_label)

            if prd_mask is None:
                continue

            iou, loss = compute_iou_loss(prd_mask, prd_scores, gt_mask)
            loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (itr + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                predictor.model.zero_grad()

            scheduler.step()
            total_iou += iou.mean().item()
            total_loss += loss.item()

    mean_iou = total_iou / train_size
    mean_loss = total_loss / train_size

    val_iou = evaluate(predictor, val_data, val_files, max_masks)
    print(f"Epoch {epoch+1}: Train IoU = {mean_iou:.4f}, Train Loss = {mean_loss:.4f}, Val IoU = {val_iou:.4f}")

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(predictor.model.state_dict(), "best_model.torch")
        print(f"New best model saved, Val IoU = {best_val_iou:.4f}")
        no_improvement_count = 0  # Reset counter when improvement is seen
    else:
        no_improvement_count += 1
        print(f"No improvement in validation IoU for {no_improvement_count} epochs.")

    # Early stopping check
    if no_improvement_count >= patience:
        print(f"Early stopping triggered. No improvement in validation IoU for {patience} consecutive epochs.")
        break
