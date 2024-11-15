import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt

from utils.testutil import (
    process_split_data, 
    load_mri_patient_data, 
    split_mri_data_by_organ,
    calculate_metrics,
    save_test_prediction,
 
)

def train_sam_decoder(
    model: torch.nn.Module,
    train_data: List[Dict],
    device: torch.device,
    save_dir: str,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
) -> torch.nn.Module:
    """Train SAM's decoder on CT data"""
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.mask_decoder.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    metrics_history = defaultdict(list)
    best_dice = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_metrics = defaultdict(list)
        
        for idx, sample in enumerate(tqdm(train_data, desc=f"Training epoch {epoch+1}/{num_epochs}")):
            
            image = torch.from_numpy(sample['frame'].astype(np.float32))
            image = image.permute(2, 0, 1).unsqueeze(0)
            image = F.interpolate(image, size=(1024, 1024), mode='bilinear', align_corners=False)
            image = (image / 255.0).to(device)
            
            
            target_mask = torch.from_numpy(sample['mask']).float()
            target_mask = F.interpolate(
                target_mask.unsqueeze(0).unsqueeze(0),
                size=(1024, 1024),
                mode='nearest'
            ).to(device)
            
            
            target_mask_small = F.interpolate(
                target_mask,
                size=(256, 256),
                mode='nearest'
            )
            
           
            box = get_bbox_from_mask(target_mask)
            box_torch = torch.as_tensor(box * 1024, device=device)
            box_torch = box_torch[None, None, :]
            
            
            with torch.no_grad():
                image_embedding = model.image_encoder(image)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None
                )
            
            mask_predictions, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            
            loss = F.binary_cross_entropy_with_logits(mask_predictions, target_mask_small)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            with torch.no_grad():
                mask_predictions_upscaled = F.interpolate(
                    mask_predictions,
                    size=(1024, 1024),
                    mode='bilinear',
                    align_corners=False
                )
                metrics = calculate_metrics(
                    torch.sigmoid(mask_predictions_upscaled),
                    target_mask
                )
            
            epoch_losses.append(loss.item())
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
                
            
            if idx % 50 == 0:
                save_test_prediction(
                    image[0],
                    target_mask[0],
                    torch.sigmoid(mask_predictions_upscaled)[0],
                    save_dir,
                    f"epoch_{epoch+1}_sample_{idx}"
                )
        
        
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, ", end='')
        print(", ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items()))
        
        
        if avg_metrics['dice'] > best_dice:
            best_dice = avg_metrics['dice']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            
        
        metrics_history['loss'].append(avg_loss)
        for k, v in avg_metrics.items():
            metrics_history[k].append(v)
            
   
    plt.figure(figsize=(10, 5))
    for metric in ['dice', 'miou']:
        plt.plot(metrics_history[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training Metrics')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    return model

def get_bbox_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Calculate bounding box from binary mask"""
   
    if mask.dim() == 4:  # [B, C, H, W]
        mask = mask.squeeze(0).squeeze(0)
    elif mask.dim() == 3:  # [C, H, W]
        mask = mask.squeeze(0)
    
    
    non_zero_coords = torch.nonzero(mask > 0)
    
    if len(non_zero_coords) == 0:
        return torch.tensor([0, 0, 1, 1], device=mask.device)
    
    
    y_min = non_zero_coords[:, 0].min()
    y_max = non_zero_coords[:, 0].max()
    x_min = non_zero_coords[:, 1].min()
    x_max = non_zero_coords[:, 1].max()
    
    h, w = mask.shape
    bbox = torch.tensor([
        x_min.float() / w,
        y_min.float() / h,
        x_max.float() / w,
        y_max.float() / h
    ], device=mask.device)
    
    return bbox

def test_sam(
    model: torch.nn.Module,
    test_data: List[Dict],
    device: torch.device,
    save_dir: str,
    data_type: str = 'CT',
    organ: str = 'liver'
) -> Dict[str, float]:
    """
    Test SAM on a dataset
    
    Args:
        model: SAM model
        test_data: List of test samples
        device: torch device
        save_dir: Directory to save results
        data_type: Type of data ('CT' or 'MRI')
        organ: Target organ for MRI data
    """
    os.makedirs(save_dir, exist_ok=True)
    metrics_log = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_data, desc=f"Testing {data_type} - {organ}")):
            
            image = torch.from_numpy(sample['frame'].astype(np.float32))
            image = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            image = F.interpolate(image, size=(1024, 1024), mode='bilinear', align_corners=False)
            image = (image / 255.0).to(device)
            
            
            if 'masks' in sample and data_type == 'MRI':
                mask = sample['masks'].get(organ, np.zeros_like(list(sample['masks'].values())[0]))
            else:
                mask = sample['mask']
            
            target_mask = torch.from_numpy(mask).float()
            target_mask = F.interpolate(
                target_mask.unsqueeze(0).unsqueeze(0),
                size=(1024, 1024),
                mode='nearest'
            ).to(device)
            
            
            box = get_bbox_from_mask(target_mask)
            box_torch = torch.as_tensor(box * 1024, device=device)
            box_torch = box_torch[None, None, :]  # [1, 1, 4]
            
            
            image_embedding = model.image_encoder(image)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None
            )
            mask_predictions, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            
            metrics = calculate_metrics(
                torch.sigmoid(mask_predictions),
                target_mask
            )
            
            for k, v in metrics.items():
                metrics_log[k].append(v)
            
           
            if idx % 20 == 0:  
                save_test_prediction(
                    image[0],
                    target_mask[0],
                    torch.sigmoid(mask_predictions)[0],
                    save_dir,
                    f"{data_type}_{organ}_sample_{idx}"
                )
    
    
    results = {k: float(np.mean(v)) for k, v in metrics_log.items()}
    
    
    results_dir = os.path.join(save_dir, f"{data_type}_{organ}_results.txt")
    with open(results_dir, 'w') as f:
        f.write(f"Results for {data_type} - {organ}:\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    model = model.to(device)
    
    
    save_dir = "sam_finetuned_results"
    os.makedirs(save_dir, exist_ok=True)
    
    
    ct_train_data = process_split_data("../../Testing_Sets/CT", ["1", "10"])
    
    
    print("Training SAM decoder on CT data...")
    model = train_sam_decoder(
        model=model,
        train_data=ct_train_data,
        device=device,
        save_dir=save_dir
    )

    mri_test_data = []
    mri_patient_ids = os.listdir("../../Testing_Sets/MR")
    for patient_id in mri_patient_ids:
        patient_dir = os.path.join("../../Testing_Sets/MR", patient_id)
        mri_test_data.extend(load_mri_patient_data(patient_dir, 'T1DUAL'))
        mri_test_data.extend(load_mri_patient_data(patient_dir, 'T2SPIR'))
    
    
    mri_splits = split_mri_data_by_organ(mri_test_data)
    
    
    mri_results = {}
    for organ in ['liver', 'kidney_r', 'kidney_l', 'spleen']:
        if organ in mri_splits and mri_splits[organ]['eval']:
            mri_results[organ] = test_sam(
                model=model,
                test_data=mri_splits[organ]['eval'],
                device=device,
                save_dir=os.path.join(save_dir, f"MRI_{organ}"),
                data_type='MRI',
                organ=organ
            )
    
    
    print("\nComparative Results:")
    with open(os.path.join(save_dir, "comparative_results.txt"), 'w') as f:
        f.write("SAM Baseline Results\n\n")
        
        
        
        # MRI Results
        for organ in ['liver', 'kidney_r', 'kidney_l', 'spleen']:
            if organ in mri_results:
                print(f"\nMRI Results ({organ}):")
                f.write(f"\nMRI Results ({organ}):\n")
                for metric, value in mri_results[organ].items():
                    print(f"  {metric}: {value:.4f}")
                    f.write(f"  {metric}: {value:.4f}\n")

if __name__ == "__main__":
    main()