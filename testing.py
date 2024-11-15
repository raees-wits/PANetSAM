
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
import pandas as pd
from utils.testutil import *




def test_with_improved_adaptation(
    model: ProtoNetSAM,
    ct_data: List[Dict],
    mri_splits: Dict[str, Dict[str, List[Dict]]],
    device: torch.device,
    save_dir: str,
    adaptation_episodes: int = 30,
    evaluation_episodes: int = 10,
    adaptation_epochs: int = 3,
    k_shot: int = 5,
    support_type: str = 'ct'
) -> Dict[str, Dict[str, float]]:

    
    os.makedirs(save_dir, exist_ok=True)
    results = defaultdict(lambda: defaultdict(list))
    
    print(f"\nTesting with {support_type.upper()} support")
    print("\nData distribution:")
    for organ, splits in mri_splits.items():
        print(f"{organ}:")
        print(f"  Adaptation set: {len(splits['adapt'])} samples")
        print(f"  Evaluation set: {len(splits['eval'])} samples")
    
    for organ, splits in mri_splits.items():
        if not splits['eval']:
            print(f"\nSkipping {organ} - no evaluation data")
            continue
            
        print(f"\nProcessing {organ}")
        
        
        # Prepare support data
        if support_type == 'ct':
            if organ != 'liver':
                print(f"Warning: Using CT liver data as support for {organ}")
            support_data = ct_data
        else:
            support_data = splits['adapt']
            
        if not support_data:
            print(f"Skipping {organ} - no support data")
            continue
            
      
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.0001
        )
        
        organ_save_dir = os.path.join(save_dir, f'{organ}_{support_type}_support')
        os.makedirs(organ_save_dir, exist_ok=True)
       
       
        
        # Adaptation phase
        model.train()
        for epoch in range(adaptation_epochs):
            metrics_log = defaultdict(list)
            scaler = torch.cuda.amp.GradScaler()
            
            for episode in tqdm(range(adaptation_episodes), 
                         desc=f"Adapting {organ} epoch {epoch+1}"):
                episode_data = create_multi_class_cross_domain_episode(
                    support_data=support_data,
                    query_data=splits['adapt'],
                    k_shot=k_shot,
                    n_query=3,
                    device=device,
                    is_support_ct=(support_type == 'ct'),
                    target_organ=organ
                )
                
                with torch.cuda.amp.autocast():
                    outputs = model(
                        support_images=episode_data['support_images'],
                        support_masks=episode_data['support_masks'],
                        query_images=episode_data['query_images'],
                        query_masks=episode_data['query_masks'],
                        use_gt_box=True
                    )
                    loss = model.compute_loss(outputs, episode_data['query_masks'])
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
         
            metrics = calculate_metrics(outputs['masks'], 
                                        episode_data['query_masks'])
           
            metrics_log['loss'].append(loss.item())
            metrics_log['dice'].append(metrics['dice'])
            metrics_log['miou'].append(metrics['miou'])
            
           
            epoch_metrics = {k: np.mean(v) for k, v in metrics_log.items()}
            print(f"Adaptation Epoch {epoch+1} - ", end='')
            print(", ".join(f"{k}: {v:.4f}" for k, v in epoch_metrics.items()))
            
        
        # Evaluation phase
        model.eval()
        eval_metrics = defaultdict(list)
        
        with torch.no_grad():
            for episode in range(evaluation_episodes):
                episode_data = create_multi_class_cross_domain_episode(
                    support_data=support_data,
                    query_data=splits['eval'],
                    k_shot=k_shot,
                    n_query=1,
                    device=device,
                    is_support_ct=(support_type == 'ct'),
                    target_organ=organ
                )
                
                outputs = model(
                    support_images=episode_data['support_images'],
                    support_masks=episode_data['support_masks'],
                    query_images=episode_data['query_images'],
                    query_masks=episode_data['query_masks'],
                    use_gt_box=True
                )
                
                metrics = calculate_metrics(
                    outputs['masks'],
                    episode_data['query_masks']
                )
                
                
                
                
                for k, v in metrics.items():
                    eval_metrics[k].append(v)
                
                
                if episode > 5:
                    save_test_prediction(
                        episode_data['query_images'][0],
                        episode_data['query_masks'][0],
                        outputs['masks'][0],
                        organ_save_dir,
                        f"ep_{episode}"
                    )
                    
                    
                    support_features, query_features, fg_protos, bg_protos = \
                        model.get_prototype_features(
                            episode_data['support_images'],
                            episode_data['support_masks'],
                            episode_data['query_images']
                        )
                    
                    prototype_save_dir = os.path.join(organ_save_dir, 'prototypes')
                    os.makedirs(prototype_save_dir, exist_ok=True)
                    
                    
                    visualize_prototypes(
                        support_features,
                        fg_protos,
                        bg_protos,
                        query_features,
                        prototype_save_dir,
                        f"ep_{episode}"
                    )
            
        
        results[f"{organ}_{support_type}_support"] = {
            k: float(np.mean(v)) for k, v in eval_metrics.items()
        }
               
        
     
        print(f"\nResults for {organ} ({support_type} support):")
        for metric, value in results[f"{organ}_{support_type}_support"].items():
            print(f"  {metric}: {value:.4f}")
    
    save_test_summary(results, save_dir)

    return results



def save_test_summary(results: Dict[str, Dict[str, float]], save_dir: str):
   
    with open(os.path.join(save_dir, 'test_results.txt'), 'w') as f:
        for domain, metrics in results.items():
            f.write(f"\n{domain.upper()} Results:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")




def load_fresh_model(model_path: str, device: torch.device) -> ProtoNetSAM:
    
    model = ProtoNetSAM(
        checkpoint_path="sam_vit_h_4b8939.pth",
        model_type="vit_h",
        finetune_mask_decoder=True
    )
    model.load_state_dict(torch.load(model_path))
    return model.to(device)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./training_outputs/best_model.pth"
    
    
    ct_test_data = process_split_data("../Testing_Sets/CT", ["1", "10", "21"])
    
   
    mri_test_data = []
    mri_patient_ids = os.listdir("../Testing_Sets/MR")
    for patient_id in mri_patient_ids:
        patient_dir = os.path.join("../Testing_Sets/MR", patient_id)
        mri_test_data.extend(load_mri_patient_data(patient_dir, 'T1DUAL'))
        mri_test_data.extend(load_mri_patient_data(patient_dir, 'T2SPIR'))
    
    
    mri_splits = split_mri_data_by_organ(mri_test_data)
    
    # Test with CT support
    print("\nRunning CT support experiments...")
    model = load_fresh_model(model_path, device)
    ct_results = test_with_improved_adaptation(
        model=model,
        ct_data=ct_test_data,
        mri_splits=mri_splits,
        device=device,
        save_dir="test_results_ct_support",
        support_type='ct'
    )
    
    # Print CT results structure for debugging
    print("\nCT Results structure:")
    for key in ct_results.keys():
        print(f"Key: {key}")
        print(f"Values: {ct_results[key]}")
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Test with MRI support
    print("\nRunning MRI support experiments...")
    model = load_fresh_model(model_path, device)
    mri_results = test_with_improved_adaptation(
        model=model,
        ct_data=ct_test_data,
        mri_splits=mri_splits,
        device=device,
        save_dir="test_results_mri_support",
        support_type='mri'
    )
    
    # Print MRI results structure for debugging
    print("\nMRI Results structure:")
    for key in mri_results.keys():
        print(f"Key: {key}")
        print(f"Values: {mri_results[key]}")
    
    # Clear GPU memory
    del model
    torch.cuda.empty_cache()
   
    print("\nComparative Results:")
    for organ in ['liver', 'kidney_r', 'kidney_l', 'spleen']:
        print(f"\n{organ.upper()}:")
        
        print("CT Support:")
        ct_key = f"{organ}_ct_support"
        if ct_key in ct_results:
            for metric, value in ct_results[ct_key].items():
                print(f"  {metric}: {value:.4f}")
        else:
            print("  No results available")
        
        print("MRI Support:")
        mri_key = f"{organ}_mri_support"
        if mri_key in mri_results:
            for metric, value in mri_results[mri_key].items():
                print(f"  {metric}: {value:.4f}")
        else:
            print("  No results available")
    
    
    with open("comparative_results.txt", "w") as f:
        f.write("Comparative Results:\n")
        for organ in ['liver', 'kidney_r', 'kidney_l', 'spleen']:
            f.write(f"\n{organ.upper()}:\n")
            
            # Write CT results
            f.write("CT Support:\n")
            ct_key = f"{organ}_ct_support"
            if ct_key in ct_results:
                for metric, value in ct_results[ct_key].items():
                    f.write(f"  {metric}: {value:.4f}\n")
            else:
                f.write("  No results available\n")
            
            # Write MRI results
            f.write("MRI Support:\n")
            mri_key = f"{organ}_mri_support"
            if mri_key in mri_results:
                for metric, value in mri_results[mri_key].items():
                    f.write(f"  {metric}: {value:.4f}\n")
            else:
                f.write("  No results available\n")

if __name__ == "__main__":
    main()