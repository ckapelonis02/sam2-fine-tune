import hydra
import numpy as np
import torch
import cv2
import os
import random
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.train_helper import read_batch
from sam2.train_helper import read_dataset
from sam2.train_helper import visualize_entry
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

predictor = SAM2ImagePredictor(sam2_model)
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)
optimizer = torch.optim.AdamW(
    params=predictor.model.parameters(),
    lr=1e-5,
    weight_decay=4e-5
)

scaler = torch.cuda.amp.GradScaler()

with open("/kaggle/input/mosaic-data-4k/sorted_ancient.txt", "r") as file:
    file_names = [int(line.strip()) for line in file]

data_size = 1000
top_files = file_names[:data_size]

random.shuffle(top_files)

data_dict = read_dataset(
    images_path="/kaggle/input/mosaic-data-4k/ancient_images",
    masks_path="/kaggle/input/mosaic-data-4k/masks",
    file_names=top_files
)

mean_iou = 0
max_masks = 75
for itr in range(100000):
    with torch.cuda.amp.autocast():
        image, masks, input_point, input_label = read_batch(data_dict, itr % data_size, max_masks)
        if (masks.shape[0] == 0):
            continue
        # visualize_entry(image, masks, input_point)

        # Segment the image using SAM
        predictor.set_image(image)  # apply SAM image encoder to the image

        # Prompt encoding
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
            )

        # Mask decoder
        batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features
            )

        # Upscale the masks to the original image resolution
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

        # Segmentation Loss calculation
        gt_mask = torch.tensor((masks / 255).astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

        # Score loss calculation (intersection over union) IoU
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05  # mix losses

        # Backpropagation
        predictor.model.zero_grad()  # empty gradient
        scaler.scale(loss).backward()  # Backpropagate
        scaler.step(optimizer)
        scaler.update()  # Mix precision

        if (itr % 500 == 0):
            torch.save(predictor.model.state_dict(), f"model{itr}.torch")
            print("Saved model.")

        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        if (itr % 100 == 0):
            print(f"step {itr} Accuracy (IoU) = {mean_iou}")

        # visualize_training_results(image, masks, prd_masks, mean_iou, itr)
