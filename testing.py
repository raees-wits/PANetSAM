
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
    Load MRI patient data for specific sequence
    Args:
        patient_dir: path to patient directory
        sequence_type: 'T1DUAL' or 'T2SPIR'
    """
    results = []
    
    # Get sequence directory
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

    # Get sorted lists of files
    dicom_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith('.dcm')])
    mask_files = sorted([f for f in os.listdir(ground_dir) if f.endswith('.png')])
    
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
        mask = np.array(Image.open(mask_path))
        mask = ((mask >= 55) & (mask <= 70)).astype(np.uint8) 
        
        if np.any(mask):
            results.append({
                'frame': frame_rgb,
                'mask': mask,
                'patient_id': os.path.basename(patient_dir),
                'sequence': sequence_type,
                'filename': dcm_file
            })
    
    return results
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
    
    plt.figtext(0.97, 0.87, 'GT', fontsize=10, ha='left', va='center')
    plt.figtext(0.97, 0.77, 'Pred', fontsize=10, ha='left', va='center')
    plt.figtext(0.97, 0.67, 'Prompt', fontsize=10, ha='left', va='center')
    
    # Add Dice score
    if torch.is_tensor(query_mask) and torch.is_tensor(pred_mask):
        dice = 2 * (pred_mask * query_mask).sum() / (pred_mask.sum() + query_mask.sum() + 1e-8)
        plt.figtext(0.97, 0.57, f'Dice: {dice:.3f}', fontsize=10, ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{identifier}.png",
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2)
    plt.close()

def create_cross_domain_episode(support_data: List[Dict],
                              query_data: List[Dict],
                              k_shot: int,
                              n_query: int,
                              device: torch.device) -> Dict[str, torch.Tensor]:
    """Create episode with support from one domain and query from another"""
    # Select random support and query samples
    if len(support_data) < k_shot or len(query_data) < n_query:
        print(f"Warning: Not enough samples. Support: {len(support_data)}, Query: {len(query_data)}")
        print(f"Required - Support: {k_shot}, Query: {n_query}")
        
        # Use random choices with replacement if necessary
        support_samples = random.choices(support_data, k=k_shot) if len(support_data) < k_shot else random.sample(support_data, k_shot)
        query_samples = random.choices(query_data, k=n_query) if len(query_data) < n_query else random.sample(query_data, n_query)
    else:
        support_samples = random.sample(support_data, k_shot)
        query_samples = random.sample(query_data, n_query)
    
    # Process support set (CT)
    support_images = []
    support_masks = []
    for sample in support_samples:
        prepared = prepare_data(sample, device)
        support_images.append(prepared['image'])
        support_masks.append(prepared['mask'])
    
    # Process query set (MRI)
    query_images = []
    query_masks = []
    for sample in query_samples:
        prepared = prepare_data(sample, device)
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


def test_model(model: ProtoNetSAM,
               ct_data: List[Dict],
               mri_data: List[Dict],
               device: torch.device,
               save_dir: str,
               num_episodes: int = 25) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive testing of the model
    Returns metrics for both same-domain and cross-domain performance
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'ct_to_ct': defaultdict(list),
        'ct_to_mri_t1': defaultdict(list),
        'ct_to_mri_t2': defaultdict(list)
    }
    
    # Split MRI data by sequence
    mri_t1_data = [d for d in mri_data if d['sequence'] == 'T1DUAL']
    mri_t2_data = [d for d in mri_data if d['sequence'] == 'T2SPIR']
    
    print(f"\nData distribution:")
    print(f"CT samples: {len(ct_data)}")
    print(f"MRI T1 samples: {len(mri_t1_data)}")
    print(f"MRI T2 samples: {len(mri_t2_data)}")
    
    with torch.no_grad():
        # 1. CT → CT Testing
        ct_save_dir = os.path.join(save_dir, 'ct_to_ct')
        os.makedirs(ct_save_dir, exist_ok=True)
        
        for episode in tqdm(range(num_episodes), desc="Testing CT→CT"):
            episode_data = create_episode(ct_data, k_shot=5, n_query=1, device=device)
            outputs = model(
                support_images=episode_data['support_images'],
                support_masks=episode_data['support_masks'],
                query_images=episode_data['query_images'],
                query_masks=episode_data['query_masks'],
                use_gt_box=True
            )
            metrics = calculate_metrics(outputs['masks'], episode_data['query_masks'])
            
            for k, v in metrics.items():
                results['ct_to_ct'][k].append(v)
            
            save_test_prediction(
                episode_data['query_images'][0],
                episode_data['query_masks'][0],
                outputs['masks'][0],
                ct_save_dir,
                f"ct_ep_{episode}"
            )
        
        # 2. CT → MRI T1 Testing
        if len(mri_t1_data) > 0:
            t1_save_dir = os.path.join(save_dir, 'ct_to_mri_t1')
            os.makedirs(t1_save_dir, exist_ok=True)
            
            for episode in tqdm(range(num_episodes), desc="Testing CT→MRI T1"):
                episode_data = create_cross_domain_episode(
                    ct_data, mri_t1_data, k_shot=5, n_query=1, device=device
                )
                outputs = model(
                    support_images=episode_data['support_images'],
                    support_masks=episode_data['support_masks'],
                    query_images=episode_data['query_images'],
                    query_masks=episode_data['query_masks'],
                    use_gt_box=True
                )
                metrics = calculate_metrics(outputs['masks'], episode_data['query_masks'])
                
                for k, v in metrics.items():
                    results['ct_to_mri_t1'][k].append(v)
                
                save_test_prediction(
                    episode_data['query_images'][0],
                    episode_data['query_masks'][0],
                    outputs['masks'][0],
                    t1_save_dir,
                    f"t1_ep_{episode}"
                )
        
        # 3. CT → MRI T2 Testing
        if len(mri_t2_data) > 0:
            t2_save_dir = os.path.join(save_dir, 'ct_to_mri_t2')
            os.makedirs(t2_save_dir, exist_ok=True)
            
            for episode in tqdm(range(num_episodes), desc="Testing CT→MRI T2"):
                episode_data = create_cross_domain_episode(
                    ct_data, mri_t2_data, k_shot=5, n_query=1, device=device
                )
                outputs = model(
                    support_images=episode_data['support_images'],
                    support_masks=episode_data['support_masks'],
                    query_images=episode_data['query_images'],
                    query_masks=episode_data['query_masks'],
                    use_gt_box=True
                )
                metrics = calculate_metrics(outputs['masks'], episode_data['query_masks'])
                
                for k, v in metrics.items():
                    results['ct_to_mri_t2'][k].append(v)
                
                save_test_prediction(
                    episode_data['query_images'][0],
                    episode_data['query_masks'][0],
                    outputs['masks'][0],
                    t2_save_dir,
                    f"t2_ep_{episode}"
                )
    
    # mean metrics
    final_results = {}
    for domain in results:
        final_results[domain] = {
            k: np.mean(v) for k, v in results[domain].items()
        }
    
   
    save_test_summary(final_results, save_dir)
    
    return final_results

def save_test_summary(results: Dict[str, Dict[str, float]], save_dir: str):
    """Save test results summary"""
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        for domain, metrics in results.items():
            f.write(f"\n{domain.upper()} Results:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./training_outputs/best_model.pth"
    test_save_dir = "test_results"
    

    ct_test_data = process_split_data("../Testing_Sets/CT", ["1", "10", "21"])
    
   
    mri_test_data = []
    for patient_id in ["1", "10", "21"]:
        patient_dir = os.path.join("../Testing_Sets/MR", patient_id)
        mri_test_data.extend(load_mri_patient_data(patient_dir, 'T1DUAL'))
        mri_test_data.extend(load_mri_patient_data(patient_dir, 'T2SPIR'))
    
   
    model = ProtoNetSAM(checkpoint_path="sam_vit_h_4b8939.pth", model_type="vit_h")
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # ===========testing=================
    results = test_model(
        model=model,
        ct_data=ct_test_data,
        mri_data=mri_test_data,
        device=device,
        save_dir=test_save_dir
    )
    
    print("\nTesting Complete! Results summary:")
    for domain, metrics in results.items():
        print(f"\n{domain.upper()}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
