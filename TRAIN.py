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

max_res = 1024

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder

optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler() # mixed precision


# Reading the data paths
data_dir = "/home/ckapelonis/Desktop/thesis/thesis-code/mosaic_generator/data/"
data = []
for name in os.listdir(data_dir + "output_images_original/"):
    data.append(
        {
        "image": data_dir + "output_images_original/" + name,
        "annotation": data_dir + "output_images_mask/" + name[:-4] + ".png"
        }
    )

def read_batch(data):
    # Choose a random image and its binary annotation mask
    ent = data[np.random.randint(len(data))]
    img = cv2.imread(ent["image"])[..., ::-1]  # Read image (RGB)
    ann_map = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)  # Read binary annotation map (0 or 255)

    # Resize image and annotation map to fit the input size
    r = np.min([max_res / img.shape[1], max_res / img.shape[0]])  
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))  
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)  

    # Initialize lists for masks and points
    masks = []
    points = []

    # Find contours (each tesserae boundary)
    contours, _ = cv2.findContours(255 - ann_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if len(contour) >= 3:
            # Create binary mask for tessera
            mask = np.zeros_like(ann_map)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            masks.append(mask)

            # Get all coordinates inside the tessera
            coords = np.argwhere(mask > 0)  
            if len(coords) > 0:
                np.random.shuffle(coords)  # Shuffle the coordinates to remove bias
                yx = coords[0]  # Pick the first (now truly random) point
                points.append([[yx[1], yx[0]]])  

    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def visualize_mask():
    # Visualize the result
    img, masks, points, _ = read_batch(data)

    # Plot the input image
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title("Input Image (Resized)")
    plt.axis("off")
    plt.show()

    # Plot the combined binary annotation mask
    plt.figure(figsize=(8, 8))
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)  
    plt.imshow(combined_mask, cmap='gray')  
    plt.title("Combined Mask (Tesserae in White)")
    plt.axis("off")
    plt.show()

    # Plot the image with truly random points inside tesserae
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    for point in points:
        plt.plot(point[0][0], point[0][1], 'ro', markersize=5)
    plt.title("Image with Randomly Distributed Points")
    plt.axis("off")
    plt.show()

for itr in range(100000):
    with torch.cuda.amp.autocast(): # cast to mix precision
        image, mask, input_point, input_label = read_batch(data)
        if (mask.shape[0] == 0):
            continue

        predictor.set_image(image) # apply SAM image encoder to the image

        # Prompt encoding
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
            )

        # Mask decoder
        batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
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

        # Segmentaion Loss caclulation
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0]) # Turn logit map to probability map
        seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

        # Score loss calculation (intersection over union) IoU
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss=seg_loss+score_loss*0.05  # mix losses

        # Apply back propogation
        predictor.model.zero_grad() # empty gradient
        scaler.scale(loss).backward()  # Backpropogate
        scaler.step(optimizer)
        scaler.update() # Mix precision

        if (itr % 100 == 0):
            torch.save(predictor.model.state_dict(), "model.torch")
            print("Saved model.")

        # Display results
        if (itr == 0):
            mean_iou = 0
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        print("step)", itr, "Accuracy (IoU) = ", mean_iou)
