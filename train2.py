# Train/Fine-Tune SAM 2 on the LabPics 1 dataset

# This script use a single image batch, if you want to train with multi image per batch check this script:
# https://github.com/sagieppel/fine-tune-train_segment_anything_2_in_60_lines_of_code/blob/main/TRAIN_multi_image_batch.py

# Toturial: https://medium.com/@sagieppel/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3
# Main repo: https://github.com/facebookresearch/segment-anything-2
# Labpics Dataset can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1
# Pretrained models for sam2 Can be downloaded from: https://github.com/facebookresearch/segment-anything-2?tab=readme-ov-file#download-checkpoints
import hydra
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configurations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module('sam2', version_base='1.2')
sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "../sam2_configs/sam2_hiera_t.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

optimizer = torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler() # mixed precision

data_dir=r"LabPicsV1/" # Path to dataset (LabPics 1)
data=[] # list of files in dataset
for ff, name in enumerate(os.listdir(data_dir+"Simple/Train/Image/")):  # go over all folder annotation
    data.append({"image":data_dir+"Simple/Train/Image/"+name,"annotation":data_dir+"Simple/Train/Instance/"+name[:-4]+".png"})


def read_batch(data): # read random image and its annotaion from  the dataset (LabPics)

   #  select image

        ent  = data[np.random.randint(len(data))] # choose random entry
        Img = cv2.imread(ent["image"])[...,::-1]  # read image
        ann_map = cv2.imread(ent["annotation"]) # read annotation

   # resize image

        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]]) # scalling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)

   # merge vessels and materials annotations

        mat_map = ann_map[:,:,0] # material annotation map
        ves_map = ann_map[:,:,2] # vessel  annotaion map
        mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1) # merge maps

   # Get binary masks and points

        inds = np.unique(mat_map)[1:] # load all indices
        points= []
        masks = []
        for ind in inds:
            mask=(mat_map == ind).astype(np.uint8) # make binary mask corresponding to index ind
            masks.append(mask)
            coords = np.argwhere(mask > 0) # get all coordinates in mask
            yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
            points.append([[yx[1], yx[0]]])
        print(len(points), len(masks))
        return Img,np.array(masks),np.array(points), np.ones([len(masks),1])

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

# for i in range(40):
#     image, masks, input_point, input_label = read_batch(data)
#     visualize_entry(image, masks, input_point)

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

# Inside your training loop:
for itr in range(100000):
    with torch.cuda.amp.autocast(): # cast to mixed precision
        image, masks, input_point, input_label = read_batch(data)
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
        gt_mask = torch.tensor(masks.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()
        print("gt_mask min:", gt_mask.min().item(), "gt_mask max:", gt_mask.max().item())


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

        if itr % 100 == 0:
            torch.save(predictor.model.state_dict(), "model.torch")
            print("Saved model.")

        # Track IoU
        if itr == 0:
            mean_iou = 0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        print(f"step {itr} Accuracy (IoU) = {mean_iou}")

        visualize_training_results(image, masks, prd_masks, mean_iou, itr)
