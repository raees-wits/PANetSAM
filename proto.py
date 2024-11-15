import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
import numpy as np

class ProtoNetSAM(nn.Module):
    def __init__(self, 
                checkpoint_path, model_type="vit_h",
                finetune_mask_decoder=False):

        super().__init__()
        # SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        # Dont train SAM encode parameters, far too many parameters even with vit_b
        for param in self.sam.parameters():
            param.requires_grad = False

        if finetune_mask_decoder:
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = True
                
        self.finetune_mask_decoder = finetune_mask_decoder
            
        # resize images to be standard for SAM input
        self.image_size = 1024
            
        #Prototype learning decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        #Prototype adaptation
        self.prototype_adaptor = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dice_loss = DiceLoss()


    def preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SAM"""
        if x.dtype == torch.uint8:
            x = x.float()
        elif x.max() <= 1.0:
            x = x * 255.0
            
        # Normalising images
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(x.device)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(x.device)
        x = (x - pixel_mean) / pixel_std
    
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return x
        
    def compute_prototypes(self, support_features, support_masks):
        """Compute class prototypes from support features"""
        B, S, C, H, W = support_features.shape
        support_features = support_features.view(B*S, C, H, W)
        
        
        support_masks = F.interpolate(
            support_masks.view(B*S, 1, *support_masks.shape[-2:]), 
            size=(H, W), 
            mode='nearest'
        )
        
       
        support_features = self.prototype_adaptor(support_features)
        
        # prototypes for support images
        fg_prototypes = []
        bg_prototypes = []
        
        for i in range(B*S):
            # masks for the support images
            curr_mask = support_masks[i:i+1]  
            curr_features = support_features[i:i+1]  
            
            # foreground prototype
            fg_mask = (curr_mask > 0).float()
            fg_proto = (curr_features * fg_mask).sum(dim=(2, 3)) / (fg_mask.sum(dim=(2, 3)) + 1e-6)
            fg_prototypes.append(fg_proto)
            
            # background prototype
            bg_mask = (curr_mask == 0).float()
            bg_proto = (curr_features * bg_mask).sum(dim=(2, 3)) / (bg_mask.sum(dim=(2, 3)) + 1e-6)
            bg_prototypes.append(bg_proto)
        
        # Stack all prototypes to be in dimension [B*S, C]
        fg_prototypes = torch.stack(fg_prototypes)  
        bg_prototypes = torch.stack(bg_prototypes)  
        
        return fg_prototypes, bg_prototypes
        
    def get_bbox_from_mask(self, mask):
        """Calculate bounding box from binary mask"""
        if mask.dim() == 4:
            # squeeze batch and channel dimensions
            mask = mask.squeeze(0).squeeze(0)  
        elif mask.dim() == 3:
            # squeeze channel dimension
            mask = mask.squeeze(0)  
            
        indices = torch.nonzero(mask > 0)
        
        if len(indices) == 0:
            return torch.tensor([0, 0, 1, 1], device=mask.device)
        
       
        y_min = indices[:, 0].min()
        y_max = indices[:, 0].max()
        x_min = indices[:, 1].min()
        x_max = indices[:, 1].max()
        
        h, w = mask.shape
       
        bbox = torch.tensor([
            x_min.float() / w,
            y_min.float() / h,
            x_max.float() / w,
            y_max.float() / h
        ], device=mask.device)
        
        return bbox
    
    def forward(self, support_images, support_masks, query_images, query_masks=None, use_gt_box=False):
        """
        Args:
            support_images: [B, S, 3, H, W]
            support_masks: [B, S, 1, H, W]
            query_images: [Q, 3, H, W]
            query_masks: [Q, 1, H, W] 
            use_gt_box: bool 
        """
        target_size = query_masks.shape[-2:] if query_masks is not None else (1024, 1024)
        original_size = support_images.shape[-2:]
        B, S = support_images.shape[:2]
        Q = query_images.shape[0]

        
        input_h, input_w = query_images.shape[-2:]
        # support and query masks resized to match target size
        support_masks = F.interpolate(
            support_masks.view(B*S, 1, *support_masks.shape[-2:]),
            size=target_size,
            mode='nearest').view(B, S, 1, *target_size)
        
        # size debugging 
        # print("Original size:", original_size)
        # print("Query image size:", query_images.shape)
        # print("Query mask size:", query_masks.shape if query_masks is not None else None)
        
        # prepare images for SAM
        support_images = support_images.view(B*S, 3, *support_images.shape[-2:])
        support_images = self.preprocess_image(support_images)
        query_images = self.preprocess_image(query_images)
        
        #Get the image embeddings 
        # where support is [B*S, 256, H', W']
        # and query is [Q, 256, H', W']
        with torch.no_grad():
            support_features = self.sam.image_encoder(support_images)  
            query_features = self.sam.image_encoder(query_images)  
        
        # prototype feature adaptor
        support_features = self.prototype_adaptor(support_features).view(B, S, -1, *support_features.shape[-2:])
        query_features = self.prototype_adaptor(query_features)
        
       
        support_masks = F.interpolate(support_masks.view(B*S, 1, *support_masks.shape[-2:]), 
                                    size=support_features.shape[-2:], 
                                    mode='nearest')
        
        #prototypes for support
        fg_prototypes = []
        bg_prototypes = []
        
        for b in range(B):
            for s in range(S):
                # feat has dim [256, H, W]
                 # mask has dim [1, H, W]
                feat = support_features[b, s]  
                mask = support_masks[b*S + s] 
                
                
                fg_mask = (mask > 0).float()
                fg_proto = (feat * fg_mask).sum(dim=(-1, -2)) / (fg_mask.sum() + 1e-6)
                
               
                bg_mask = (mask == 0).float()
                bg_proto = (feat * bg_mask).sum(dim=(-1, -2)) / (bg_mask.sum() + 1e-6)
                
                fg_prototypes.append(fg_proto)
                bg_prototypes.append(bg_proto)
        
        # prototype dim : [B*S, 256]
        fg_prototypes = torch.stack(fg_prototypes)  
        bg_prototypes = torch.stack(bg_prototypes)  
        
        # query similarity maps - q_feat has dim  [256, H, W]
        similarity_maps = []
        for q in range(Q):
            q_feat = query_features[q]  
            
            
            fg_sims = []
            bg_sims = []
            
            for p in range(len(fg_prototypes)):
                # [256, 1, 1]
                fg_proto = fg_prototypes[p].view(-1, 1, 1)  
                bg_proto = bg_prototypes[p].view(-1, 1, 1) 
                
                fg_sim = F.cosine_similarity(q_feat.unsqueeze(0), 
                                           fg_proto.unsqueeze(0), dim=1).squeeze(0)
                bg_sim = F.cosine_similarity(q_feat.unsqueeze(0), 
                                           bg_proto.unsqueeze(0), dim=1).squeeze(0)
                
                fg_sims.append(fg_sim)
                bg_sims.append(bg_sim)
            
            #taking the average similarity of prototypes - [H, W]
            fg_sim = torch.stack(fg_sims).mean(0)  
            bg_sim = torch.stack(bg_sims).mean(0)  
            
            similarity_maps.append((fg_sim - bg_sim))
        
        # need to reshape to [Q, 1, H, W]
        similarity_maps = torch.stack(similarity_maps).unsqueeze(1)  
        
        # SAM predictions - without bounding box = self prompting based off the similarity maps
        if use_gt_box and query_masks is not None:
            boxes = [self.get_bbox_from_mask(mask) for mask in query_masks]
        else:
            temp_pred = F.interpolate(similarity_maps, size=query_images.shape[-2:], 
                                    mode='bilinear', align_corners=False)
            pred_masks = (temp_pred > 0).float()
            boxes = [self.get_bbox_from_mask(mask) for mask in pred_masks]
        
        sam_masks = []
        for i, box in enumerate(boxes):
            box_np = box.cpu().numpy()
            box_torch = torch.as_tensor(box_np * query_images.shape[-1], device=box.device)
            box_torch = box_torch[None, None, :]
            
            if not self.finetune_mask_decoder:
                with torch.no_grad():
                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None
                    )
                    
                    mask_logits, _ = self.sam.mask_decoder(
                        image_embeddings=query_features[i:i+1],
                        image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
            else:
                # fine-tune gradients for mask decoder
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None
                )
                
                mask_logits, _ = self.sam.mask_decoder(
                    image_embeddings=query_features[i:i+1],
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                
            
            sam_masks.append(mask_logits.squeeze())
        
        # masks dim changed from  [Q, H, W] -> [Q, 1, H, W]
        sam_masks = torch.stack(sam_masks, dim=0).squeeze(1)  
        sam_masks = sam_masks.unsqueeze(1)  
        
        
        h, w = sam_masks.shape[-2:]
        similarity_maps = F.interpolate(similarity_maps, size=(h, w), 
                                      mode='bilinear', align_corners=False)
        
        # shapes for debugging
        # print("similarity_maps shape:", similarity_maps.shape)
        # print("sam_masks shape:", sam_masks.shape)
        
        # Combined predictions masks -  [Q, 2, H, W]
        combined_features = torch.cat([similarity_maps, sam_masks], dim=1)  
        logits = self.decoder(combined_features)
        
        # logits to match mask sizes
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        masks = torch.sigmoid(logits)
        
        return {
            'masks': torch.sigmoid(logits),
            'logits': logits
        }
        

    def get_prototype_features(self, support_images, support_masks, query_images):
        """Extract and return prototype features for visualization"""
        # Preprocess images
        support_images = support_images.view(-1, 3, *support_images.shape[-2:])
        support_images = self.preprocess_image(support_images)
        query_images = self.preprocess_image(query_images)
        
        # Get image embeddings
        with torch.no_grad():
            support_features = self.sam.image_encoder(support_images)
            query_features = self.sam.image_encoder(query_images)
        
        # Apply prototype adaptor
        support_features = self.prototype_adaptor(support_features)
        query_features = self.prototype_adaptor(query_features)
        
        # Compute prototypes
        fg_prototypes, bg_prototypes = self.compute_prototypes(
            support_features.view(1, -1, *support_features.shape[-3:]),
            support_masks
        )
    
        return support_features, query_features, fg_prototypes, bg_prototypes
    
    def compute_loss(self, outputs, targets):
        """Compute combined loss"""
        
        assert outputs['logits'].shape == targets.shape, \
            f"Shape mismatch: {outputs['logits'].shape} vs {targets.shape}"
            
        dice_loss = self.dice_loss(outputs['logits'], targets)
        bce_loss = F.binary_cross_entropy_with_logits(outputs['logits'], targets)
        return dice_loss + bce_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        """
        Args:
            logits: [N, 1, H, W]
            targets: [N, 1, H, W]
        """
        assert logits.shape == targets.shape, \
            f"Shape mismatch in DiceLoss: {logits.shape} vs {targets.shape}"
            
        probs = torch.sigmoid(logits)
        
        # using batch and channel dimensions, flatten height and width
        flat_probs = probs.view(probs.size(0), probs.size(1), -1)
        flat_targets = targets.view(targets.size(0), targets.size(1), -1)
        
        intersection = (flat_probs * flat_targets).sum(-1)
        union = flat_probs.sum(-1) + flat_targets.sum(-1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()