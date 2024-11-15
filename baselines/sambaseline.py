import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
import numpy as np

class BaselineSAM(nn.Module):
    def __init__(self, 
                checkpoint_path, model_type="vit_h",
                finetune_mask_decoder=False):

        super().__init__()
        # Initialize SAM model
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        # Freeze all SAM parameters
        for param in self.sam.parameters():
            param.requires_grad = False

        # Optionally fine-tune the mask decoder
        if finetune_mask_decoder:
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = True

        self.finetune_mask_decoder = finetune_mask_decoder

        # Define image size for SAM input
        self.image_size = 1024

        # Initialize Dice Loss
        self.dice_loss = DiceLoss()

    def preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess images for SAM."""
        if x.dtype == torch.uint8:
            x = x.float()
        elif x.max() <= 1.0:
            x = x * 255.0

        # Normalize images
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(x.device)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(x.device)
        x = (x - pixel_mean) / pixel_std

        # Resize images
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        return x

    def get_bbox_from_mask(self, mask):
        """Calculate bounding box from binary mask."""
        if mask.dim() == 4:
            # Squeeze batch and channel dimensions
            mask = mask.squeeze(0).squeeze(0)  
        elif mask.dim() == 3:
            # Squeeze channel dimension
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

    def forward(self, query_images, query_masks=None, use_gt_box=False):
        """
        Args:
            query_images: [Q, 3, H, W]
            query_masks: [Q, 1, H, W] (Optional)
            use_gt_box: bool (Whether to use ground truth bounding boxes)
        """
        # Determine target size for masks
        target_size = query_masks.shape[-2:] if query_masks is not None else (1024, 1024)
        Q = query_images.shape[0]

        # Preprocess query images
        query_images = self.preprocess_image(query_images)

        # Obtain image embeddings from SAM's image encoder
        with torch.no_grad():
            query_features = self.sam.image_encoder(query_images)  

        sam_masks = []
        for q in range(Q):
            if use_gt_box and query_masks is not None:
                # Compute bounding box from ground truth mask
                box = self.get_bbox_from_mask(query_masks[q])
                box_np = box.cpu().numpy()
                box_torch = torch.as_tensor(box_np * query_images.shape[-1], device=box.device)
                box_torch = box_torch[None, None, :]
            else:
                box_torch = None  # No bounding box provided

            if not self.finetune_mask_decoder:
                with torch.no_grad():
                    # Encode prompts (bounding boxes)
                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=None,
                        boxes=box_torch,
                        masks=None
                    )

                    # Decode masks
                    mask_logits, _ = self.sam.mask_decoder(
                        image_embeddings=query_features[q:q+1],
                        image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
            else:
                # Fine-tune mask decoder with gradient flow
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None
                )

                mask_logits, _ = self.sam.mask_decoder(
                    image_embeddings=query_features[q:q+1],
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

            sam_masks.append(mask_logits.squeeze())

        # Stack masks and adjust dimensions
        sam_masks = torch.stack(sam_masks, dim=0).unsqueeze(1)  

        # Resize masks to target size
        masks = F.interpolate(sam_masks, size=target_size, mode='bilinear', align_corners=False)

        return {
            'masks': torch.sigmoid(masks),
            'logits': masks
        }

    def compute_loss(self, outputs, targets):
        """Compute combined Dice and Binary Cross-Entropy loss."""
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

        # Flatten the tensors
        flat_probs = probs.view(probs.size(0), probs.size(1), -1)
        flat_targets = targets.view(targets.size(0), targets.size(1), -1)

        # Compute intersection and union
        intersection = (flat_probs * flat_targets).sum(-1)
        union = flat_probs.sum(-1) + flat_targets.sum(-1)

        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
