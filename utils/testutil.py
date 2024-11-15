import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from collections import defaultdict
import pydicom
from PIL import Image
from proto import ProtoNetSAM
import random

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
    
    # Prepare support and query sets
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
    
    support_images = torch.cat(support_images, dim=0).unsqueeze(0)  # [1, S, 3, H, W]
    support_masks = torch.cat(support_masks, dim=0).unsqueeze(0)    # [1, S, 1, H, W]
    query_images = torch.cat(query_images, dim=0)                   # [Q, 3, H, W]
    query_masks = torch.cat(query_masks, dim=0)                     # [Q, 1, H, W]
    
    return {
        'support_images': support_images,
        'support_masks': support_masks,
        'query_images': query_images,
        'query_masks': query_masks
    }

def prepare_data(slice_data: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Prepare single slice data"""
    # Prepare image - normalize to [0,1]
    frame = slice_data['frame'].astype(np.float32)
    frame = frame / 255.0  # Normalize to [0,1]
    frame = torch.from_numpy(frame).permute(2, 0, 1)  # [3, H, W]
    
    # No need for interpolation here as SAM processor will handle resizing
    frame = frame.unsqueeze(0)  # Add batch dimension [1, 3, H, W]
    
    # Prepare mask
    mask = torch.from_numpy(slice_data['mask']).float()
    
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                        size=(1024, 1024), 
                        mode='nearest')
    
    return {
        'image': frame.to(device),
        'mask': mask.to(device)
    }

def process_split_data(data_dir: str, patient_ids: List[str]) -> List[Dict]:
    """Process data for specific patients"""
    all_results = []
    
    for patient_id in patient_ids:
        patient_dir = os.path.join(data_dir, patient_id)
        if os.path.isdir(patient_dir):
            patient_results = load_patient_data(patient_dir)
            all_results.extend(patient_results)
    
    return all_results


def load_patient_data(patient_dir: str) -> List[Dict]:
    """Load patient DICOM and ground truth data from CHAOS dataset"""
    results = []
    
    # Get DICOM directory and Ground directory
    dicom_dir = os.path.join(patient_dir, 'DICOM_anon')
    ground_dir = os.path.join(patient_dir, 'Ground')
    
    if not (os.path.exists(dicom_dir) and os.path.exists(ground_dir)):
        return results

    # Get sorted lists of files
    dicom_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith('.dcm')])
    mask_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.png')])
    
    for dcm_file, mask_file in zip(dicom_files, mask_files):
        # Load DICOM image
        dcm_path = os.path.join(dicom_dir, dcm_file)
        ds = pydicom.dcmread(dcm_path)
        frame = ds.pixel_array.astype(float)
        
        # Normalize DICOM image to [0,1] range
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = (frame * 255).astype(np.uint8)
        
        # Convert to RGB (SAM expects 3 channels)
        frame_rgb = np.stack([frame] * 3, axis=-1)
        
        # Load mask
        mask_path = os.path.join(ground_dir, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # Convert mask to binary (any value > 0 represents liver)
        # print(f"CT mask unique values before binarization: {np.unique(mask)}")
        mask = (mask > 0).astype(np.uint8)
        
        # Only include slices with liver annotations
        if np.any(mask):
            results.append({
                'frame': frame_rgb,
                'mask': mask,
                'patient_id': os.path.basename(patient_dir)
            })
    
    return results



def load_mri_patient_data(patient_dir: str, sequence_type: str = 'T1DUAL') -> List[Dict]:
    """
    Load MRI patient data with multiple organ classes
    Args:
        patient_dir: path to patient directory
        sequence_type: 'T1DUAL' or 'T2SPIR'
    """
    results = []
    
    sequence_dir = os.path.join(patient_dir, sequence_type)
    if not os.path.exists(sequence_dir):
        return results
        
    # For T1DUAL, use InPhase images
    if sequence_type == 'T1DUAL':
        dicom_dir = os.path.join(sequence_dir, 'DICOM_anon', 'InPhase')
    else:
        dicom_dir = os.path.join(sequence_dir, 'DICOM_anon')
        
    ground_dir = os.path.join(sequence_dir, 'Ground')
    
    if not (os.path.exists(dicom_dir) and os.path.exists(ground_dir)):
        return results

    dicom_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith('.dcm')])
    mask_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.png')])
    
    # Define organ class ranges in the CHAOS dataset
    ORGAN_RANGES = {
        'liver': (55, 70),      # Liver range
        'kidney_r': (125, 140), # Right kidney range
        'kidney_l': (145, 160), # Left kidney range
        'spleen': (175, 190),   # Spleen range
    }
    
    for dcm_file, mask_file in zip(dicom_files, mask_files):
        # Load DICOM image
        dcm_path = os.path.join(dicom_dir, dcm_file)
        ds = pydicom.dcmread(dcm_path)
        frame = ds.pixel_array.astype(float)
        
        # Normalize DICOM image
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = (frame * 255).astype(np.uint8)
        frame_rgb = np.stack([frame] * 3, axis=-1)
        
        # Load mask
        mask_path = os.path.join(ground_dir, mask_file)
        full_mask = np.array(Image.open(mask_path))
        
        # Create separate binary mask for each organ
        organ_masks = {}
        for organ, (min_val, max_val) in ORGAN_RANGES.items():
            organ_mask = ((full_mask >= min_val) & (full_mask <= max_val)).astype(np.uint8)
            if np.any(organ_mask):  # Only include if organ is present
                organ_masks[organ] = organ_mask
        
        if organ_masks:  # If any organ is present
            results.append({
                'frame': frame_rgb,
                'masks': organ_masks,
                'patient_id': os.path.basename(patient_dir),
                'sequence': sequence_type,
                'filename': dcm_file
            })
    
    return results

def split_mri_data_by_organ(mri_data: List[Dict], 
                          adaptation_ratio: float = 0.5) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Split MRI data by organ into adaptation and evaluation sets
    Args:
        mri_data: List of MRI data dictionaries
        adaptation_ratio: Ratio of data to use for adaptation
    Returns:
        Dictionary with splits by organ and sequence type
    """
    # Initialize data structure
    splits = {
        'liver': {'adapt': [], 'eval': []},
        'kidney_r': {'adapt': [], 'eval': []},
        'kidney_l': {'adapt': [], 'eval': []},
        'spleen': {'adapt': [], 'eval': []}
    }
    
    # Group by patient ID first to avoid data leakage
    patient_data = defaultdict(list)
    for item in mri_data:
        patient_data[item['patient_id']].append(item)
    
    # Split patients into adaptation and evaluation sets
    patient_ids = list(patient_data.keys())
    random.shuffle(patient_ids)
    split_idx = int(len(patient_ids) * adaptation_ratio)
    
    adaptation_patients = set(patient_ids[:split_idx])
    evaluation_patients = set(patient_ids[split_idx:])
    
    # Distribute data by organ while maintaining patient separation
    for patient_id, samples in patient_data.items():
        target_split = 'adapt' if patient_id in adaptation_patients else 'eval'
        
        for sample in samples:
            for organ in ['liver', 'kidney_r', 'kidney_l', 'spleen']:
                if organ in sample['masks']:
                    # Transform to match expected format
                    transformed_sample = {
                        'frame': sample['frame'],
                        'mask': sample['masks'][organ],
                        'patient_id': sample['patient_id'],
                        'sequence': sample['sequence']
                    }
                    splits[organ][target_split].append(transformed_sample)
    
    return splits



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
    # Ensure same size
    if pred_masks.shape != target_masks.shape:
        pred_masks = F.interpolate(pred_masks, size=target_masks.shape[-2:], 
                                 mode='bilinear', align_corners=False)
    
    # Threshold predictions
    pred_masks = (pred_masks > threshold).float()
    
    # Calculate Dice score
    dice_score = 2 * (pred_masks * target_masks).sum() / \
                (pred_masks.sum() + target_masks.sum() + 1e-8)
    
    # Calculate IoU for each image in batch
    ious = []
    for i in range(pred_masks.shape[0]):
        pred = pred_masks[i].squeeze()  # [H, W]
        target = target_masks[i].squeeze()  # [H, W]
        
        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection  # Subtract intersection to avoid counting it twice
        
        # Calculate IoU
        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou)
    
    # Calculate mean IoU
    miou = torch.stack(ious).mean()
    
    return {
        'dice': dice_score.item(),
        'miou': miou.item()
    }



def save_test_prediction(
    query_image: torch.Tensor,
    query_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    save_path: str,
    identifier: str  # Changed from episode to identifier for more flexibility
):
    """
    Save predictions with GT prompt box visualization
    Args:
        query_image: [3, H, W]
        query_mask: [1, H, W]
        pred_mask: [1, H, W]
        save_path: str
        identifier: str (can be episode number or any other identifier)
    """
    plt.figure(figsize=(15, 5))
    
    # Process query image
    img = query_image.squeeze().cpu().permute(1, 2, 0).numpy()
    h, w = img.shape[:2]
    
    # Resize masks to match query image size
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
    
    # Get prompt box from GT mask
    prompt_box = get_bbox_from_mask(query_mask)
    
    # Convert normalized bbox coordinates to pixel coordinates
    box_pixels = [
        int(prompt_box[0] * w),   # x1
        int(prompt_box[1] * h),   # y1
        int(prompt_box[2] * w),   # x2
        int(prompt_box[3] * h)    # y2
    ]
    
    # Define colors
    gt_color = (0.18, 0.8, 0.44)  # Green
    pred_color = (0.90, 0.30, 0.24)  # Red
    box_color = 'yellow'  # Prompt box color
    
    # Create color maps
    gt_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'gt',
        [(0, 0, 0, 0), gt_color + (1.0,)]
    )
    pred_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'pred',
        [(0, 0, 0, 0), pred_color + (1.0,)]
    )
    
    # Original image
    plt.subplot(131)
    plt.imshow(img)
    plt.title('Query Image', fontsize=12, pad=10)
    plt.axis('off')
    
    # Ground truth with prompt box
    plt.subplot(132)
    plt.imshow(img)
    plt.imshow(query_mask.squeeze().cpu().numpy(),
              alpha=0.7,
              cmap=gt_cmap)
    
    # Add prompt box
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
    
    # Prediction
    plt.subplot(133)
    plt.imshow(img)
    plt.imshow(pred_mask.squeeze().cpu().numpy()> 0.5,
              alpha=0.7,
              cmap=pred_cmap)
    plt.title('Prediction', fontsize=12, pad=10)
    plt.axis('off')
    
    # Add color reference boxes and labels
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
    
    plt.figtext(0.97, 0.87, 'GT', fontsize=10, ha='left', va='center', color="white")
    plt.figtext(0.97, 0.77, 'Pred', fontsize=10, ha='left', va='center',color="white")
    plt.figtext(0.97, 0.67, 'Prompt', fontsize=10, ha='left', va='center',color="white")
    

    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{identifier}.png",
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

def create_multi_class_cross_domain_episode(
    support_data: List[Dict],  # CT or MRI data
    query_data: List[Dict],    # MRI data (multiple organs)
    k_shot: int,
    n_query: int,
    device: torch.device,
    is_support_ct: bool = True,  # Flag to indicate if support data is CT
    target_organ: str = 'liver',  # Specify which organ to focus on
    target_size: Tuple[int, int] = (1024, 1024)
) -> Dict[str, torch.Tensor]:
    """Create episode with CT/MRI support and multi-class MRI query"""
    
    # Select random support samples
    support_samples = random.sample(support_data, k_shot) if len(support_data) >= k_shot else random.choices(support_data, k=k_shot)
    
    # Select random query samples
    query_samples = random.sample(query_data, n_query) if len(query_data) >= n_query else random.choices(query_data, k=n_query)
    
    # Process support set
    support_images = []
    support_masks = []
    
    for sample in support_samples:
        # Prepare frame
        frame = sample['frame'].astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # [3, H, W]
        
        # Resize image
        frame = F.interpolate(
            frame.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Get appropriate mask
        if is_support_ct:
            mask = sample['mask']
        else:
            mask = sample['mask']  # Already transformed to single organ mask
            
        # Prepare mask
        mask_tensor = torch.from_numpy(mask).float()
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='nearest'
        )
        
        support_images.append(frame.unsqueeze(0).to(device))
        support_masks.append(mask_tensor.to(device))
    
    # Process query set
    query_images = []
    query_masks = []
    
    for sample in query_samples:
        # Prepare image
        frame = sample['frame'].astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # [3, H, W]
        
        # Resize image
        frame = F.interpolate(
            frame.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        query_images.append(frame.unsqueeze(0).to(device))
        
        # Get appropriate mask based on whether it's multi-organ or single organ
        if 'masks' in sample:  # Multi-organ MRI format
            if target_organ in sample['masks']:
                mask = sample['masks'][target_organ]
            else:
                # If target organ not present, use zero mask
                mask = np.zeros_like(sample['masks'][list(sample['masks'].keys())[0]])
        else:  # Single organ format
            mask = sample['mask']
        
        # Prepare mask
        mask_tensor = torch.from_numpy(mask).float()
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='nearest'
        )
        query_masks.append(mask_tensor.to(device))
    
    # Stack tensors
    support_images = torch.cat(support_images, dim=0).unsqueeze(0)  # [1, S, 3, H, W]
    support_masks = torch.cat(support_masks, dim=0).unsqueeze(0)    # [1, S, 1, H, W]
    query_images = torch.cat(query_images, dim=0)                   # [Q, 3, H, W]
    query_masks = torch.cat(query_masks, dim=0)                     # [Q, 1, H, W]
    
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
def visualize_prototypes(
    support_features: torch.Tensor,  # [B*S, C, H, W]
    fg_prototypes: torch.Tensor,     # [B*S, C]
    bg_prototypes: torch.Tensor,     # [B*S, C]
    query_features: torch.Tensor,    # [Q, C, H, W]  
    save_dir: str,
    identifier: str
):
    """Visualize prototype activations and similarities"""
    plt.figure(figsize=(15, 10))
    
    # Get feature map dimensions
    _, C, H, W = query_features.shape
    n_prototypes = fg_prototypes.shape[0]
    
    # Compute similarity maps for each prototype
    similarity_maps = []
    for i in range(n_prototypes):
        fg_proto = fg_prototypes[i].view(-1, 1, 1)  # [C, 1, 1]
        bg_proto = bg_prototypes[i].view(-1, 1, 1)  # [C, 1, 1]
        
        # Compute similarity for each query feature
        fg_sims = []
        bg_sims = []
        for q in range(query_features.shape[0]):
            q_feat = query_features[q]  # [C, H, W]
            
            # Compute cosine similarity
            fg_sim = F.cosine_similarity(
                q_feat.unsqueeze(0),
                fg_proto.unsqueeze(0),
                dim=1
            ).squeeze(0)  # [H, W]
            
            bg_sim = F.cosine_similarity(
                q_feat.unsqueeze(0),
                bg_proto.unsqueeze(0),
                dim=1
            ).squeeze(0)  # [H, W]
            
            fg_sims.append(fg_sim)
            bg_sims.append(bg_sim)
        
        # Average similarities across query samples
        avg_fg_sim = torch.stack(fg_sims).mean(0)  # [H, W]
        avg_bg_sim = torch.stack(bg_sims).mean(0)  # [H, W]
        
        similarity_maps.append({
            'fg': avg_fg_sim.cpu().numpy(),
            'bg': avg_bg_sim.cpu().numpy(),
            'diff': (avg_fg_sim - avg_bg_sim).cpu().numpy()
        })
    
    # Plot similarity maps
    n_cols = 3  # fg, bg, diff
    n_rows = n_prototypes
    
    for i in range(n_prototypes):
        # Foreground similarity
        plt.subplot(n_rows, n_cols, i * n_cols + 1)
        plt.imshow(similarity_maps[i]['fg'], cmap='hot')
        plt.title(f'Proto {i+1} FG Sim')
        plt.colorbar()
        plt.axis('off')
        
        # Background similarity
        plt.subplot(n_rows, n_cols, i * n_cols + 2)
        plt.imshow(similarity_maps[i]['bg'], cmap='hot')
        plt.title(f'Proto {i+1} BG Sim')
        plt.colorbar()
        plt.axis('off')
        
        # Difference map
        plt.subplot(n_rows, n_cols, i * n_cols + 3)
        plt.imshow(similarity_maps[i]['diff'], cmap='seismic')
        plt.title(f'Proto {i+1} Diff')
        plt.colorbar()
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/prototypes_{identifier}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()