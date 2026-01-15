"""
Cascaded Blending Module - Implements true cascaded blending: layer-by-layer blending,
where blended features serve as input to the next layer.

Core functionalities:
1. Record feature snapshots when editing mask in GUI
2. Create SynthesisNetwork that supports cascaded blending
3. Generate blended images (masked regions remain unchanged, non-masked regions change normally)
"""

import torch
import torch.nn.functional as F
import copy
from typing import List, Optional, Tuple


def prepare_blending_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Prepare blending mask
    
    Args:
        mask: Original mask, shape [H, W] or [1, H, W], 0 for editable region, 1 for non-editable region
        device: Target device
        
    Returns:
        blending_mask: Mask for blending, shape [1, 1, H, W]
    """
    if mask is None:
        return None
    
    mask_tensor = mask.to(device)
    
    # Ensure correct dimensions
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(1)  # [B, 1, H, W]
    
    # Ensure float type
    if mask_tensor.dtype != torch.float32:
        mask_tensor = mask_tensor.float()
    
    return mask_tensor


def record_mask_snapshot(renderer, trunc_psi: float = 0.7) -> List[torch.Tensor]:
    """
    Record feature map snapshots when editing mask
    
    Called when editing mask in GUI, saves current feature maps for subsequent blending
    
    Args:
        renderer: Renderer object
        trunc_psi: Truncation psi value
        
    Returns:
        snapshot_features: List of feature map snapshots for each layer
    """
    G = renderer.G
    ws = renderer.w
    
    # Prepare ws (consistent with logic in core.py)
    if ws.dim() == 2:
        ws = ws.unsqueeze(1).repeat(1, 6, 1)
    ws = torch.cat([ws[:, :6, :], renderer.w0[:, 6:, :]], dim=1)
    
    label = torch.zeros([1, G.c_dim], device=renderer._device)
    
    # Get current feature maps (without gradient computation)
    with torch.no_grad():
        _, features = G(ws, label, truncation_psi=trunc_psi,
                       noise_mode='const', input_is_w=True, return_feature=True)
    
    # Save feature map snapshots (deep copy)
    snapshot_features = []
    for feat in features:
        snapshot_features.append(feat.detach().clone())
    
    print(f"[core_blending] Recorded {len(snapshot_features)} layer feature map snapshots for cascaded blending")
    return snapshot_features


class CascadedBlendingSynthesisNetwork(torch.nn.Module):
    """
    SynthesisNetwork that supports cascaded blending
    
    Layer-by-layer blending: blended features from one layer serve as input to the next layer
    Implements true cascaded blending effect
    """
    
    def __init__(self, original_synthesis, snapshot_features, mask, blend_coeffs, start_idx=3):
        super().__init__()
        # Copy all attributes
        self.w_dim = original_synthesis.w_dim
        self.img_resolution = original_synthesis.img_resolution
        self.img_resolution_log2 = original_synthesis.img_resolution_log2
        self.img_channels = original_synthesis.img_channels
        self.num_fp16_res = original_synthesis.num_fp16_res
        self.block_resolutions = original_synthesis.block_resolutions
        self.num_ws = original_synthesis.num_ws
        
        # Copy all blocks (deep copy to avoid modifying original network)
        for res in self.block_resolutions:
            block = getattr(original_synthesis, f'b{res}')
            # Create deep copy of block
            setattr(self, f'b{res}', copy.deepcopy(block))
        
        # Blending related parameters
        self.snapshot_features = snapshot_features  # List of recorded feature maps
        self.mask = mask  # Mask region [1, 1, H, W]
        self.blend_coeffs = blend_coeffs  # List of blending coefficients for each layer
        self.start_idx = start_idx  # Block index to start blending
    
    def forward(self, ws, return_feature=False, **block_kwargs):
        block_ws = []
        features = []
        
        # Split ws (same as original logic)
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv
        
        x = img = None
        for block_idx, (res, cur_ws) in enumerate(zip(self.block_resolutions, block_ws)):
            block = getattr(self, f'b{res}')
            
            # Execute block forward propagation
            x, img = block(x, img, cur_ws, **block_kwargs)
            
            # Cascaded blending logic: layer-by-layer blending, blended features serve as input to next layer
            if (block_idx >= self.start_idx and 
                self.snapshot_features is not None and 
                block_idx < len(self.snapshot_features) and
                self.mask is not None and x is not None):
                
                # Get blending coefficient
                alpha = self.blend_coeffs[block_idx] if block_idx < len(self.blend_coeffs) else 0.5
                
                # Resize mask to match current feature map
                _, _, h, w = x.shape
                mask_resized = F.interpolate(
                    self.mask,
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                )
                mask_resized = (mask_resized > 0.5).float()
                
                # Get corresponding snapshot feature
                snap_feat = self.snapshot_features[block_idx]
                
                # Ensure snap_feat is on correct device and shape matches
                if snap_feat.shape != x.shape:
                    snap_feat = F.interpolate(snap_feat, size=(h, w), mode='bilinear', align_corners=False)
                
                # Alpha blending (only in mask region)
                # Mask region: blend current feature and snapshot feature
                # Non-mask region: keep current feature unchanged
                blended = alpha * x + (1 - alpha) * snap_feat
                x = torch.where(mask_resized > 0.5, blended, x)
                # Important: x is now blended feature, will serve as input to next block
            
            features.append(x)
        
        if return_feature:
            return img, features
        else:
            return img


def apply_cascaded_blending(
    G,
    ws: torch.Tensor,
    snapshot_features: List[torch.Tensor],
    mask: torch.Tensor,
    label: Optional[torch.Tensor] = None,
    truncation_psi: float = 0.7,
    noise_mode: str = 'const',
    start_idx: int = 6,  # Default start from layer 6 to avoid affecting gradient-computed layers
    blend_coeffs: Optional[List[float]] = None,
    **kwargs
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Apply cascaded blending to generator forward propagation
    
    Create SynthesisNetwork that supports cascaded blending, execute blended forward propagation
    
    Args:
        G: Generator network
        ws: Latent code
        snapshot_features: List of feature map snapshots
        mask: Blending mask, shape [1, 1, H, W], 1 for fixed region, 0 for editable region
        label: Conditional label
        truncation_psi: Truncation psi value
        noise_mode: Noise mode
        start_idx: Layer index to start blending (default 6, to avoid affecting gradient-computed layers)
        blend_coeffs: List of blending coefficients for each layer (must be provided by caller)
        
    Returns:
        img: Generated image
        blended_features: List of blended feature maps
    """
    if mask is None or snapshot_features is None:
        # If no mask or snapshot features, execute normal forward propagation
        with torch.no_grad():
            img, features = G(ws, label, truncation_psi=truncation_psi,
                            noise_mode=noise_mode, input_is_w=True, return_feature=True, **kwargs)
        return img, features
    
    # Prepare label
    if label is None:
        label = torch.zeros([1, G.c_dim], device=ws.device)
    
    # Validate blending coefficients
    if blend_coeffs is None:
        raise ValueError("blend_coeffs must be provided by caller")
    
    if len(blend_coeffs) != len(snapshot_features):
        raise ValueError(f"blend_coeffs length ({len(blend_coeffs)}) must match snapshot_features length ({len(snapshot_features)})")
    
    print(f"[core_blending] Applying cascaded blending: starting from layer {start_idx}")
    print(f"[core_blending] Blending coefficients: {blend_coeffs[start_idx:]}")
    
    # Save original synthesis network
    original_synthesis = G.synthesis
    
    # Create cascaded blending synthesis network
    blending_synthesis = CascadedBlendingSynthesisNetwork(
        original_synthesis=original_synthesis,
        snapshot_features=snapshot_features,
        mask=mask,
        blend_coeffs=blend_coeffs,
        start_idx=start_idx
    )
    
    # Temporarily replace synthesis network
    G.synthesis = blending_synthesis
    
    try:
        # Execute forward propagation with cascaded blending (without gradient computation)
        with torch.no_grad():
            result = G(ws, label, truncation_psi=truncation_psi,
                      noise_mode=noise_mode, input_is_w=True, return_feature=True, **kwargs)
    finally:
        # Restore original synthesis network
        G.synthesis = original_synthesis
    
    return result


def generate_blended_image(
    G,
    ws: torch.Tensor,
    snapshot_features: List[torch.Tensor],
    mask: torch.Tensor,
    label: Optional[torch.Tensor] = None,
    trunc_psi: float = 0.7,
    noise_mode: str = 'const',
    start_idx: int = 6,
    blend_coeffs: Optional[List[float]] = None,
    **kwargs
) -> torch.Tensor:
    """
    Generate cascaded blended image (simplified interface)
    
    Args:
        G: Generator network
        ws: Latent code
        snapshot_features: List of feature map snapshots
        mask: Blending mask
        label: Conditional label
        trunc_psi: Truncation psi value
        noise_mode: Noise mode
        start_idx: Layer index to start blending
        blend_coeffs: List of blending coefficients for each layer
        
    Returns:
        blended_img: Blended image [1, 3, H, W]
    """
    img, _ = apply_cascaded_blending(
        G, ws, snapshot_features, mask, label,
        truncation_psi=trunc_psi, noise_mode=noise_mode,
        start_idx=start_idx, blend_coeffs=blend_coeffs, **kwargs
    )
    return img
