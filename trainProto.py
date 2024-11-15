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

from utils.trainutil import *

def train_epoch(model: ProtoNetSAM,
                train_data: List[Dict],
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                num_episodes: int = 15) -> Dict[str, float]:
    """Training epoch"""
    model.train()
    metrics_log = defaultdict(list)
    scaler = torch.amp.GradScaler()
    
    for _ in tqdm(range(num_episodes), desc="Training"):
        episode_data = create_episode(
            train_data,
            k_shot=5,  
            n_query=3,
            device=device
        )
        # remove anything cache before forward pass, this was dying on higher kshot values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        #automatic  mixed precision
        with torch.amp.autocast('cuda'):
            outputs = model(
                support_images=episode_data['support_images'],
                support_masks=episode_data['support_masks'],
                query_images=episode_data['query_images'],
                query_masks=episode_data['query_masks'],
                use_gt_box=False
            )
            
            loss = model.compute_loss(outputs, episode_data['query_masks'])
        
        
        
        #backprop
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        #metrics to log
        metrics = calculate_metrics(
            outputs['masks'],
            episode_data['query_masks']
        )
        
        
        metrics_log['loss'].append(loss.item())
        metrics_log['dice'].append(metrics['dice'])
        metrics_log['miou'].append(metrics['miou'])
    
    return {k: np.mean(v) for k, v in metrics_log.items()}

def validate(model: ProtoNetSAM,
            val_data: List[Dict],
            device: torch.device,
            save_dir: str,
            num_episodes: int = 5) -> Dict[str, float]:
    """Validation"""
    model.eval()
    metrics_log = defaultdict(list)

    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for episode in tqdm(range(num_episodes), desc="Validating"):
            episode_data = create_episode(
                val_data,
                k_shot=5,
                n_query=1,
                device=device
            )
           
            query_masks = F.interpolate(
                episode_data['query_masks'],
                size=(1024, 1024),
                mode='nearest'
            )
            
            outputs = model(
                support_images=episode_data['support_images'],
                support_masks=episode_data['support_masks'],
                query_images=episode_data['query_images'],
                query_masks = query_masks,
                use_gt_box=False  
            )
            
           
            loss = model.compute_loss(outputs, episode_data['query_masks'])
            #NOTE: outputs are pushed through a sigmoid in the model before being returned
            pred_masks = outputs['masks']
            target_masks = episode_data['query_masks']
            
            
            metrics = calculate_metrics(
                outputs['masks'],
                episode_data['query_masks']
            )
            
            metrics_log['loss'].append(loss.item())
            metrics_log['dice'].append(metrics['dice'])
            metrics_log['miou'].append(metrics['miou'])

            
            
            save_validation_prediction(
                episode_data['query_images'][0],
                query_masks[0],
                pred_masks[0],
                save_dir,
                episode
            )
    
    return {k: np.mean(v) for k, v in metrics_log.items()}

def main():
   
    data_dir = "../Train_Sets/CT"  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "training_outputs"
    os.makedirs(save_dir, exist_ok=True)
    
    
    patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    num_val = int(len(patient_dirs) * 0.2)
    val_patients = np.random.choice(patient_dirs, size=num_val, replace=False)
   
    val_data = process_split_data(data_dir, val_patients)

    train_pool = [p for p in patient_dirs if p not in val_patients]
    
    # model initialisation - base model : sam_vit_b_01ec64, huge model : sam_vit_h_4b8939
    model = ProtoNetSAM(
        checkpoint_path="sam_vit_h_4b8939.pth",
        model_type="vit_h",
        finetune_mask_decoder = True
    ).to(device)
        
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # =============Training loop==============================
    num_epochs = 15
    best_val_dice = 0
    metrics_history = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_save_dir, exist_ok=True)
       
        #  sample subset of training patients for each epoch
        epoch_train_size = min(len(train_pool), 100)  
        epoch_train_patients = np.random.choice(
            train_pool,
            size=epoch_train_size,
            replace=False
        )
        
        
        train_data = process_split_data(data_dir, epoch_train_patients)
        train_metrics = train_epoch(model, train_data, optimizer, device)
        metrics_history['train'].append(train_metrics)
        
        # ===================Validation======================
        val_metrics = validate(
            model, 
            val_data, 
            device,
            os.path.join(save_dir, f'epoch_{epoch+1}')
        )
        metrics_history['val'].append(val_metrics)
        
        scheduler.step()
        
        
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        
    
        plt.figure(figsize=(12, 6))
        for split in ['train', 'val']:
            for metric in ['dice', 'miou']:
                values = [m[metric] for m in metrics_history[split]]
                plt.plot(values, label=f'{split}_{metric}')
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.close()
        
      
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, "
              f"mIoU: {train_metrics['miou']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, "
              f"mIoU: {val_metrics['miou']:.4f}")

if __name__ == "__main__":
    main()