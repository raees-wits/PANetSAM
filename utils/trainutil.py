
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from proto import ProtoNetSAM
import random
from collections import defaultdict
import pydicom
from PIL import Image

def load_patient_data(patient_dir: str) -> List[Dict]:
    """Load patient DICOM and ground truth data from CHAOS dataset"""
    results = []
    
   
    dicom_dir = os.path.join(patient_dir, 'DICOM_anon')
    ground_dir = os.path.join(patient_dir, 'Ground')
    
    if not (os.path.exists(dicom_dir) and os.path.exists(ground_dir)):
        return results

   
    dicom_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith('.dcm')])
    mask_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.png')])
    
    for dcm_file, mask_file in zip(dicom_files, mask_files):
       
        dcm_path = os.path.join(dicom_dir, dcm_file)
        ds = pydicom.dcmread(dcm_path)
        frame = ds.pixel_array.astype(float)
        
       
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = (frame * 255).astype(np.uint8)
        
        
        frame_rgb = np.stack([frame] * 3, axis=-1)
        
        
        mask_path = os.path.join(ground_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        
        mask = (mask > 0).astype(np.uint8)
        
        if np.any(mask):
            results.append({
                'frame': frame_rgb,
                'mask': mask,
                'patient_id': os.path.basename(patient_dir)
            })
    
    return results

def process_split_data(data_dir: str, patient_ids: List[str]) -> List[Dict]:
    """Process data for specific patients"""
    all_results = []
    
    for patient_id in patient_ids:
        patient_dir = os.path.join(data_dir, patient_id)
        if os.path.isdir(patient_dir):
            patient_results = load_patient_data(patient_dir)
            all_results.extend(patient_results)
    
    return all_results

def prepare_data(slice_data: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare single slice data"""
   
    frame = slice_data['frame'].astype(np.float32)
    frame = frame / 255.0  
    # reshape to be used with SAM [3, H, W]
    frame = torch.from_numpy(frame).permute(2, 0, 1)  
    
    # SAM handles the resizing, add batch dim: [1, 3, H, W]
    frame = frame.unsqueeze(0)  
    
   
    mask = torch.from_numpy(slice_data['mask']).float()
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                        size=(1024, 1024), 
                        mode='nearest')
    
    return {
        'image': frame.to(device),
        'mask': mask.to(device)
    }

def create_episode(data: List[Dict],
                  k_shot: int,
                  n_query: int,
                  device: torch.device) -> Dict[str, torch.Tensor]:
    """Create episode for binary liver segmentation"""
    required_samples = k_shot + n_query
    
    if len(data) < required_samples:
        selected_samples = random.choices(data, k=required_samples)
    else:
        selected_samples = random.sample(data, required_samples)
    
    support_images = []
    support_masks = []
    query_images = []
    query_masks = []
    
    for i in range(k_shot):
        prepared = prepare_data(selected_samples[i], device)
        support_images.append(prepared['image'])
        support_masks.append(prepared['mask'])
    
    for i in range(n_query):
        prepared = prepare_data(selected_samples[k_shot + i], device)
        query_images.append(prepared['image'])
        query_masks.append(prepared['mask'])
    
    # [1, S, 3, H, W]
    support_images = torch.cat(support_images, dim=0).unsqueeze(0)  
    # [1, S, 1, H, W]
    support_masks = torch.cat(support_masks, dim=0).unsqueeze(0)   
     # [Q, 3, H, W] 
    query_images = torch.cat(query_images, dim=0)   
     # [Q, 1, H, W]               
    query_masks = torch.cat(query_masks, dim=0)                    
    
    return {
        'support_images': support_images,
        'support_masks': support_masks,
        'query_images': query_images,
        'query_masks': query_masks
    }

def get_bbox_from_mask( mask: torch.Tensor) -> torch.Tensor:
    """Calculate bounding box from binary mask"""
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    
    y_indices, x_indices = torch.where(mask > 0)
    
    if len(y_indices) == 0:
        return torch.tensor([0, 0, 1, 1], device=mask.device)
    
    x1 = x_indices.min()
    y1 = y_indices.min()
    x2 = x_indices.max()
    y2 = y_indices.max()
    
    h, w = mask.shape
    bbox = torch.tensor([x1/w, y1/h, x2/w, y2/h], device=mask.device)
    
    return bbox

def save_validation_prediction(
                             query_image: torch.Tensor,
                             query_mask: torch.Tensor,
                             pred_mask: torch.Tensor,
                             save_path: str,
                             episode: int):
    """
    Save validation predictions with GT prompt box visualization
    Args:
        query_image: [3, H, W]
        query_mask: [1, H, W]
        pred_mask: [1, H, W]
        save_path: str
        episode: int
    """
    plt.figure(figsize=(15, 5))
    
    
    img = query_image.squeeze().cpu().permute(1, 2, 0).numpy()
    h, w = img.shape[:2]
    
    
    query_mask = F.interpolate(
        query_mask.unsqueeze(0),
        size=img.shape[:2],
        mode='nearest'
    ).squeeze(0)
    
    pred_mask = F.interpolate(
        pred_mask.unsqueeze(0),
        size=img.shape[:2],
        mode='nearest'
    ).squeeze(0)
    
   
    prompt_box = get_bbox_from_mask(query_mask)
    
    
    box_pixels = [
        int(prompt_box[0] * w),   
        int(prompt_box[1] * h),   
        int(prompt_box[2] * w),   
        int(prompt_box[3] * h)    
    ]
    
   
    gt_color = (0.18, 0.8, 0.44)  # -> green
    pred_color = (0.90, 0.30, 0.24)  # -> red
    box_color = 'yellow'  # -> box color
    
    
    gt_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'gt',
        [(0, 0, 0, 0), gt_color + (1.0,)]
    )
    pred_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'pred',
        [(0, 0, 0, 0), pred_color + (1.0,)]
    )
    
    # original image
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Query Image', fontsize=12, pad=10)
    plt.axis('off')
    
    # GT + prompt box
    plt.subplot(132)
    plt.imshow(img)
    plt.imshow(query_mask.squeeze().cpu().numpy(),
              alpha=0.7,
              cmap=gt_cmap)
    
    # plot the box
    rect = plt.Rectangle(
        (box_pixels[0], box_pixels[1]),
        box_pixels[2] - box_pixels[0],
        box_pixels[3] - box_pixels[1],
        fill=False,
        color=box_color,
        linewidth=2,
        linestyle='--',
        label='Prompt Box'
    )
    plt.gca().add_patch(rect)
    plt.title('Ground Truth + Prompt Box', fontsize=12, pad=10)
    plt.axis('off')
    
    # predicted mask
    plt.subplot(133)
    plt.imshow(img)
    plt.imshow(pred_mask.squeeze().cpu().numpy()>0.5,
              alpha=0.7,
              cmap=pred_cmap)
    plt.title('Prediction', fontsize=12, pad=10)
    plt.axis('off')
    
 
    gt_patch = plt.Rectangle((0.92, 0.85), 0.04, 0.04,
                           facecolor=gt_color + (0.7,),
                           transform=plt.gcf().transFigure)
    pred_patch = plt.Rectangle((0.92, 0.75), 0.04, 0.04,
                             facecolor=pred_color + (0.7,),
                             transform=plt.gcf().transFigure)
    box_patch = plt.Rectangle((0.92, 0.65), 0.04, 0.04,
                            facecolor='none',
                            edgecolor=box_color,
                            linestyle='--',
                            transform=plt.gcf().transFigure)
    
    plt.gcf().patches.extend([gt_patch, pred_patch, box_patch])
    
    plt.figtext(0.97, 0.87, 'GT', fontsize=10, ha='left', va='center')
    plt.figtext(0.97, 0.77, 'Pred', fontsize=10, ha='left', va='center')
    plt.figtext(0.97, 0.67, 'Prompt', fontsize=10, ha='left', va='center')
    
   
    if torch.is_tensor(query_mask) and torch.is_tensor(pred_mask):
        dice = 2 * (pred_mask * query_mask).sum() / (pred_mask.sum() + query_mask.sum() + 1e-8)
        plt.figtext(0.97, 0.57, f'Dice: {dice:.3f}', fontsize=10, ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/val_pred_episode_{episode}.png",
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

def calculate_metrics(pred_masks: torch.Tensor, target_masks: torch.Tensor, threshold: float = 0.5):
    """
    Calculate Dice and mIoU metrics
    Args:
        pred_masks: [N, 1, H, W] tensor of predicted probabilities
        target_masks: [N, 1, H, W] tensor of ground truth binary masks
        threshold: float, threshold for converting predictions to binary
    Returns:
        dict containing dice and miou scores
    """
   
    if pred_masks.shape != target_masks.shape:
        pred_masks = F.interpolate(pred_masks, size=target_masks.shape[-2:], 
                                 mode='bilinear', align_corners=False)
    
    
    pred_masks = (pred_masks > threshold).float()
    
    
    dice_score = 2 * (pred_masks * target_masks).sum() / \
                (pred_masks.sum() + target_masks.sum() + 1e-8)
    
   
    ious = []
    for i in range(pred_masks.shape[0]):
        # [H, W]
        pred = pred_masks[i].squeeze()  
        target = target_masks[i].squeeze()  
        
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection  
        
        
        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou)
    
    
    miou = torch.stack(ious).mean()
    
    return {
        'dice': dice_score.item(),
        'miou': miou.item()
    }
