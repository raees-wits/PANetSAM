import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from segment_anything import sam_model_registry

from utils.testutil import (
    process_split_data, 
    load_mri_patient_data, 
    split_mri_data_by_organ,
    calculate_metrics,
    save_test_prediction,
 
)

def get_bbox_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Calculate bounding box from binary mask"""
    
    if mask.dim() == 4:  # [B, C, H, W]
        mask = mask.squeeze(0).squeeze(0)
    elif mask.dim() == 3:  # [C, H, W]
        mask = mask.squeeze(0)
    
    
    non_zero_coords = torch.nonzero(mask > 0)
    
    if len(non_zero_coords) == 0:
        return torch.tensor([0, 0, 1, 1], device=mask.device)
    
    # Get min and max coordinates
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

def test_medsam(
    model: torch.nn.Module,
    test_data: List[Dict],
    device: torch.device,
    save_dir: str,
    data_type: str = 'CT',
    organ: str = 'liver'
) -> Dict[str, float]:
    """
    Test MedSAM on a dataset
    
    Args:
        model: MedSAM model
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
    
    
    model = sam_model_registry["vit_b"](checkpoint="medsam_vit_b.pth")
    model = model.to(device)
    
    
    save_dir = "medsam_test_results"
    os.makedirs(save_dir, exist_ok=True)
    
    
    
    # Load and test MRI data
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
            mri_results[organ] = test_medsam(
                model=model,
                test_data=mri_splits[organ]['eval'],
                device=device,
                save_dir=os.path.join(save_dir, f"MRI_{organ}"),
                data_type='MRI',
                organ=organ
            )
    
    
    print("\nComparative Results:")
    with open(os.path.join(save_dir, "comparative_results.txt"), 'w') as f:
        f.write("MedSAM Baseline Results\n\n")
        
        
        
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