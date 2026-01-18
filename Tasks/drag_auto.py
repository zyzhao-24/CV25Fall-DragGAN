import os
import sys
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz.renderer import Renderer
from core import render_drag_impl
from blending import record_mask_snapshot, apply_cascaded_blending, prepare_blending_mask

try:
    import cv2
except Exception:
    cv2 = None

def ensure_uint8_hwc(img):
    """
    Convert img (PIL / numpy / torch) to uint8 numpy array with shape (H, W, 3).
    Accepts:
      - PIL.Image
      - numpy: (H,W), (H,W,3), (H,W,4), (3,H,W), (1,3,H,W), etc.
      - torch: (H,W), (H,W,3), (3,H,W), (1,3,H,W), (1,H,W,3), etc.
    """
    import numpy as np
    import torch
    from PIL import Image

    # 1) PIL -> numpy RGB
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr

    # 2) torch -> numpy
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        # remove batch dim if present
        while t.dim() >= 4 and t.shape[0] == 1:
            t = t.squeeze(0)

        # Now handle common layouts
        if t.dim() == 3:
            # CHW -> HWC
            if t.shape[0] in (1, 3, 4) and t.shape[0] != t.shape[-1]:
                t = t.permute(1, 2, 0)
        elif t.dim() == 2:
            # HW grayscale
            t = t.unsqueeze(-1)
        else:
            # weird shape, try flattening safely
            t = t.squeeze()

            if t.dim() == 2:
                t = t.unsqueeze(-1)
            elif t.dim() == 3 and t.shape[0] in (1,3,4):
                t = t.permute(1,2,0)

        arr = t.numpy()

    else:
        arr = np.asarray(img)

    # 3) numpy sanitize dims
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        # If CHW, convert to HWC
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))

        # Drop alpha if RGBA
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

    # 4) dtype/range normalize to uint8
    if arr.dtype != np.uint8:
        # If float-like and looks like [0,1] or [-1,1], map to [0,255]
        if np.issubdtype(arr.dtype, np.floating):
            vmin, vmax = float(arr.min()), float(arr.max())
            if vmax <= 1.0 and vmin >= 0.0:
                arr = (arr * 255.0).round()
            elif vmax <= 1.0 and vmin >= -1.0:
                arr = ((arr * 0.5 + 0.5) * 255.0).round()
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Final assert-like cleanup
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Image has invalid shape after conversion: {arr.shape}, dtype={arr.dtype}")

    return arr


def to_tensor_img(np_img, device):
    """Convert numpy HxWx3 uint8 to torch [1,3,H,W]"""
    t = torch.from_numpy(np_img.astype(np.float32)).permute(2,0,1).unsqueeze(0)
    return t.to(device)

def to_uint8_np(img_tensor):
    """Convert torch [1,3,H,W] to numpy HxWx3 uint8"""
    if isinstance(img_tensor, torch.Tensor):
        arr = img_tensor.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    else:
        arr = np.array(img_tensor)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr

def create_mask(img_shape, mask_type='center_circle'):
    """Create mask for drag region.
    
    Args:
        img_shape: (H, W) shape
        mask_type: 'center_circle', 'center_rect', 'all_ones'
        
    Returns:
        mask: HxW with values in [0, 1]
    """
    H, W = img_shape
    mask = np.ones((H, W), dtype=np.float32)
    cy, cx = H // 2, W // 2
    
    if mask_type == 'center_circle':
        radius = min(H, W) // 4
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask[dist <= radius] = 0.0
    elif mask_type == 'center_rect':
        h_half, w_half = H // 4, W // 4
        mask[cy - h_half:cy + h_half, cx - w_half:cx + w_half] = 0.0
    elif mask_type == 'all_ones':
        mask = np.zeros((H, W), dtype=np.float32)
    else:
        mask = np.zeros((H, W), dtype=np.float32)
    
    print(f"  Mask created: type={mask_type}, area={mask.sum():.0f}, shape={mask.shape}")
    return mask

def generate_drag_points(img, mask, num_points=3):
    """Generate drag points within mask region with better spacing.
    
    Args:
        img: HxWx3 image
        mask: HxW mask
        num_points: number of drag points to generate
        
    Returns:
        handle_points: list of (y, x) starting positions
        target_points: list of (y, x) target positions
    """
    H, W = img.shape[:2]
    handle_points = []
    target_points = []
    
    # Find points in mask region
    mask_coords = np.where(mask > 0.5)
    if len(mask_coords[0]) == 0:
        print("Warning: No valid points in mask region")
        return [], []
    
    rng = np.random.RandomState(42)
    min_distance = 40  # Ensure minimum separation
    
    attempts = 0
    max_attempts = 500
    
    while len(handle_points) < num_points and attempts < max_attempts:
        attempts += 1
        
        # Pick random point in mask
        idx = rng.randint(0, len(mask_coords[0]))
        py, px = float(mask_coords[0][idx]), float(mask_coords[1][idx])
        
        # Check if it's far enough from existing points
        too_close = False
        for (epy, epx) in handle_points:
            dist = np.sqrt((py - epy) ** 2 + (px - epx) ** 2)
            if dist < min_distance:
                too_close = True
                break
        if too_close:
            continue
        
        # Random displacement (larger range: -20 to 20 pixels)
        dy = rng.randint(-40, 41)
        dx = rng.randint(-40, 41)
        ty = np.clip(py + dy, 0, H - 1)
        tx = np.clip(px + dx, 0, W - 1)
        
        # Ensure target is still in mask region and displacement is substantial
        if mask[int(ty), int(tx)] < 0.5 and (abs(dy) > 20 or abs(dx) > 25):
            handle_points.append([py, px])
            target_points.append([ty, tx])
            print(f"  Point {len(handle_points)}: handle=({py:.1f}, {px:.1f}), target=({ty:.1f}, {tx:.1f}), delta=({dy}, {dx})")
    
    if len(handle_points) < num_points:
        print(f"Warning: Only generated {len(handle_points)}/{num_points} points (max attempts reached)")
    
    print(f"  Total generated {len(handle_points)} drag points")
    return handle_points, target_points

def extract_images_from_renderer(renderer, res, mask_tensor=None):
    """
    Render and return image as numpy uint8.

    Important:
      - If you want cascaded blending to be reflected in the returned image,
        you MUST pass mask_tensor (HxW, values in {0,1}).
      - This function calls render_drag_impl directly (not renderer._render_drag_impl),
        so the cascaded blending block inside render_drag_impl can run.
    """
    # Use the same core rendering function so blending logic is not bypassed.
    render_drag_impl(
        renderer, res,
        is_drag=False,
        mask=mask_tensor,
        to_pil=True
    )

    img = res.image
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    elif isinstance(img, torch.Tensor):
        img = to_uint8_np(img)
    else:
        img = np.asarray(img).astype(np.uint8)

    return img


def compute_patch_similarity(img1, img2, mask, patch_size=5):
    """Compute patch-based similarity between two images in mask region.
    
    Using LPIPS-like perceptual loss (simplified version with VGG features).
    
    Args:
        img1: HxWx3 uint8
        img2: HxWx3 uint8
        mask: HxW mask
        patch_size: patch size for local comparison
        
    Returns:
        similarity: float in [0, 1] (higher = more similar)
    """
    try:
        from torchvision import models, transforms
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load VGG16
        vgg = models.vgg16(pretrained=True).features[:16].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Preprocess
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Extract features
        feat1 = vgg(preprocess(Image.fromarray(img1)).unsqueeze(0).to(device))
        feat2 = vgg(preprocess(Image.fromarray(img2)).unsqueeze(0).to(device))
        
        # Compute L2 distance
        feat_dist = F.l1_loss(feat1, feat2).item()
        # Normalize to [0, 1] range (higher values = larger distance)
        similarity = max(0, 1.0 - feat_dist / 10.0)  # Rough normalization
        
        return float(similarity)
    except Exception as e:
        print(f"  Warning: Feature-based similarity failed: {e}")
        # Fallback: pixel-wise L2 in mask region
        img1_f = img1.astype(np.float32) / 255.0
        img2_f = img2.astype(np.float32) / 255.0
        mask_3d = mask[:, :, np.newaxis]
        diff = np.sqrt(np.mean(((img1_f - img2_f) ** 2) * mask_3d))
        similarity = max(0, 1.0 - diff)  # Rough normalization
        return float(similarity)


from PIL import ImageDraw

def draw_point_pairs(img, handle_points, target_points,
                     r=5, draw_line=True,
                     handle_color=(255, 0, 0),   # red
                     target_color=(0, 128, 255), # blue
                     line_color=(255, 255, 0)):  # yellow
    """
    Draw handle/target point pairs on an image.

    Args:
        img: HxWx3 uint8 numpy OR PIL.Image
        handle_points: list of [y, x]
        target_points: list of [y, x]
        r: circle radius
        draw_line: whether to draw line from handle->target
    Returns:
        annotated PIL.Image (RGB)
    """
    if isinstance(img, Image.Image):
        im = img.convert("RGB").copy()
    else:
        im = Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB")

    draw = ImageDraw.Draw(im)

    n = min(len(handle_points), len(target_points))
    for i in range(n):
        hy, hx = handle_points[i]
        ty, tx = target_points[i]
        hx_i, hy_i = int(round(hx)), int(round(hy))
        tx_i, ty_i = int(round(tx)), int(round(ty))

        # Optional: line
        if draw_line:
            draw.line([(hx_i, hy_i), (tx_i, ty_i)], fill=line_color, width=2)

        # handle (red)
        draw.ellipse(
            [(hx_i - r, hy_i - r), (hx_i + r, hy_i + r)],
            outline=handle_color, width=2
        )
        draw.ellipse(
            [(hx_i - 1, hy_i - 1), (hx_i + 1, hy_i + 1)],
            fill=handle_color
        )

        # target (blue)
        draw.ellipse(
            [(tx_i - r, ty_i - r), (tx_i + r, ty_i + r)],
            outline=target_color, width=2
        )
        draw.ellipse(
            [(tx_i - 1, ty_i - 1), (tx_i + 1, ty_i + 1)],
            fill=target_color
        )

        # label
        draw.text((hx_i + r + 2, hy_i - r - 2), f"H{i}", fill=handle_color)
        draw.text((tx_i + r + 2, ty_i - r - 2), f"T{i}", fill=target_color)

    return im


def compute_matching_points_distance(handle_points, target_points, img1, img2, mask):
    """Compute average distance between matching points in mask region.
    
    Strategy: Track handle_points from img1 to img2 using optical flow or template matching.
    
    Args:
        handle_points: list of (y, x) in img1
        target_points: list of (y, x) targets for drag
        img1: HxWx3 uint8 original
        img2: HxWx3 uint8 after drag
        mask: HxW mask
        
    Returns:
        avg_distance: average L2 distance of tracked points
        distances: list of individual distances
    """
    if len(handle_points) == 0:
        return float('nan'), []
    
    distances = []
    
    # Simple template matching: for each handle_point, find best match in img2 within mask region
    for i, (hy, hx) in enumerate(handle_points):
        hy, hx = int(round(hy)), int(round(hx))
        
        # Extract template from img1 (small patch around handle point)
        patch_size = 11
        y1 = max(0, hy - patch_size // 2)
        y2 = min(img1.shape[0], hy + patch_size // 2 + 1)
        x1 = max(0, hx - patch_size // 2)
        x2 = min(img1.shape[1], hx + patch_size // 2 + 1)
        
        template = img1[y1:y2, x1:x2]
        if template.size == 0:
            continue
        
        # Try to match in img2 using sum of squared differences
        best_dist = float('inf')
        best_y, best_x = hy, hx
        
        # Search in a region around the target point
        search_range = 20
        ty, tx = int(round(target_points[i][0])), int(round(target_points[i][1]))
        
        for sy in range(max(0, ty - search_range), min(img2.shape[0], ty + search_range + 1)):
            for sx in range(max(0, tx - search_range), min(img2.shape[1], tx + search_range + 1)):
                if mask[sy, sx] < 0.5:  # Skip if outside mask
                    continue
                
                sy1 = sy - patch_size // 2
                sy2 = sy + patch_size // 2 + 1
                sx1 = sx - patch_size // 2
                sx2 = sx + patch_size // 2 + 1
                
                if sy1 < 0 or sy2 > img2.shape[0] or sx1 < 0 or sx2 > img2.shape[1]:
                    continue
                
                patch2 = img2[sy1:sy2, sx1:sx2]
                if patch2.shape != template.shape:
                    continue
                
                # SSD
                ssd = np.sum((template.astype(np.float32) - patch2.astype(np.float32)) ** 2)
                if ssd < best_dist:
                    best_dist = ssd
                    best_y, best_x = sy, sx
        
        # Distance from target to found position
        dist = np.sqrt((best_y - ty) ** 2 + (best_x - tx) ** 2)
        distances.append(dist)
        print(f"    Point {i}: target=({ty}, {tx}), matched=({best_y}, {best_x}), dist={dist:.2f}")
    
    if len(distances) == 0:
        return float('nan'), []
    
    avg_dist = float(np.mean(distances))
    return avg_dist, distances

def run_drag_eval(
    model_pkl,
    seed=0,
    out_dir='./eval_out',
    num_drag_points=2,
    drag_iterations=20,
    mask_type='center_circle',
    use_cascaded_blending=False,
    override_handle_points=None,
    override_target_points=None
):

    """Run drag evaluation with mask.
    
    Args:
        model_pkl: Path to generator model
        seed: Random seed
        out_dir: Output directory
        num_drag_points: Number of drag points (1-4)
        drag_iterations: Optimization iterations for each drag step
        mask_type: Type of mask ('center_circle', 'center_rect', 'all_ones')
        use_cascaded_blending: Whether to apply cascaded blending after drag optimization
    """
    num_drag_points = max(1, min(4, num_drag_points))  # Clamp to [1, 4]
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"=== Drag Evaluation with Mask ===")
    print(f"Model: {model_pkl}")
    print(f"Seed: {seed}, Mask type: {mask_type}, Num drag points: {num_drag_points}")
    print(f"Cascaded blending: {'Enabled' if use_cascaded_blending else 'Disabled'}")
    
    # Load generator
    print("\nStep 1: Loading generator...")
    renderer = Renderer(disable_timing=True)
    res = SimpleNamespace()
    try:
        renderer.init_network(res, pkl=model_pkl, w0_seed=seed, w_plus=True)
    except Exception as e:
        print(f"Error loading generator: {e}")
        traceback.print_exc()
        raise
    
    # Generate image 1
    print("Step 2: Generating image 1...")
    img1 = extract_images_from_renderer(renderer, res)
    print(f"  Image shape: {img1.shape}")
    Image.fromarray(img1).save(os.path.join(out_dir, 'image1.png'))
    
    # Create mask
    print("Step 3: Creating mask...")
    h, w = img1.shape[:2]
    mask = create_mask((h, w), mask_type=mask_type)
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(out_dir, 'mask.png'))
    
    # Generate drag points
    # Step 4: drag points
    if override_handle_points is not None:
        handle_points = override_handle_points
        target_points = override_target_points
        print(f"  Using benchmark points: {len(handle_points)} pairs")
    else:
        handle_points, target_points = generate_drag_points(
            img1, mask, num_points=num_drag_points
        )

        # Save visualization on original image
    print("Step 4.1: Saving point-pair visualization on original image...")
    vis1 = draw_point_pairs(img1, handle_points, target_points, r=6, draw_line=True)
    vis1.save(os.path.join(out_dir, "image1_points.png"))

    if len(handle_points) == 0:
        print("Error: No valid drag points generated")
        return None
    
    # Save points
    points_json = {
        'handle_points': handle_points,
        'target_points': target_points
    }
    with open(os.path.join(out_dir, 'drag_points.json'), 'w') as f:
        json.dump(points_json, f, indent=2)
    
    # Perform drag
    print(f"Step 5: Performing drag optimization ({drag_iterations} iterations)...")
    res_drag = SimpleNamespace()
    renderer.init_network(res_drag, pkl=model_pkl, w0_seed=seed, w_plus=True)
    # Clone and detach w to create a leaf variable (required for requires_grad)
    renderer.w = renderer.w.clone().detach().requires_grad_(True)
    renderer.w_optim = torch.optim.Adam([renderer.w], lr=0.002)
    
    # Convert mask to torch tensor
    device = renderer._device
    mask_tensor = torch.from_numpy(mask).to(device)
    
    # Record feature snapshots for cascaded blending
    if use_cascaded_blending:
        print("Step 5.0: Recording feature snapshots for cascaded blending...")
        snapshot_features = record_mask_snapshot(renderer, trunc_psi=0.7)
        renderer.mask_snapshot_features = snapshot_features
        renderer.mask_snapshot_image = snapshot_features[0]
    
    # Track losses per iteration for debugging
    loss_history = []
    
    print(f"[eval_drag_with_mask] Starting drag with handle_points={handle_points}, target_points={target_points}")
    
    # Run drag loop
    for step in range(drag_iterations):
        try:
            points_array = np.array(handle_points, dtype=np.int32)
            targets_array = np.array(target_points, dtype=np.int32)
            print(f"[eval_drag_with_mask Step {step}] Points array shape: {points_array.shape}, Targets array shape: {targets_array.shape}")
            
            render_drag_impl(
                renderer, res_drag,
                points=points_array,
                targets=targets_array,
                mask=mask_tensor,
                lambda_mask=10,
                is_drag=True,
                reset=(step == 0),
                to_pil=False
            )
            
            # Track gradient magnitude for debugging
            if renderer.w.grad is not None:
                grad_norm = float(torch.norm(renderer.w.grad).item())
                loss_history.append(grad_norm)
            
            if step % max(1, drag_iterations // 5) == 0 or step == drag_iterations - 1:
                status = f"Iteration {step}/{drag_iterations}"
                if renderer.w.grad is not None:
                    status += f", grad_norm={grad_norm:.6f}"
                print(f"  {status}")
            if step == drag_iterations - 1:
                img2 = ensure_uint8_hwc(res_drag.image)
        except Exception as e:
            print(f"  Warning: Drag step {step} failed: {e}")
            traceback.print_exc()
            break
    
    moved_points = getattr(res_drag, "points", handle_points)  # moved handle points after last iteration
    if moved_points is None:
        moved_points = handle_points

    # Extract image 2
    # print("Step 6: Extracting result image...")
    # img2 = extract_images_from_renderer(renderer, res_drag)
    
    # If cascaded blending was applied at each step, img2 is already the blended result
    # Otherwise, it's the raw dragged result
    if use_cascaded_blending:
        print("  (Result includes cascaded blending applied at each iteration)")
        Image.fromarray(img2).save(os.path.join(out_dir, 'image2_blended.png'))
    else:
        Image.fromarray(img2).save(os.path.join(out_dir, 'image2.png'))
    
    # Save visualization on result image
    print("Step 6.1: Saving point-pair visualization on result image...")
    vis2 = draw_point_pairs(img2, moved_points, target_points, r=6, draw_line=True)
    if use_cascaded_blending:
        vis2.save(os.path.join(out_dir, "image2_blended_points.png"))
    else:
        vis2.save(os.path.join(out_dir, "image2_points.png"))
    
    results = {
        'seed': seed,
        'mask_type': mask_type,
        'num_drag_points': len(handle_points),
        'drag_iterations': drag_iterations,
        'use_cascaded_blending': use_cascaded_blending,
        'output_images': {
            'image1_original': 'image1_original.png',
            'image2_dragged': 'image2_dragged_blended.png' if use_cascaded_blending else 'image2_dragged.png',
            'mask': 'mask.png'
        },
        'drag_points': points_json
    }
    
    with open(os.path.join(out_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to {os.path.join(out_dir, 'eval_results.json')}")
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate DragGAN with mask-based evaluation')
    parser.add_argument('--model', required=True, help='Path to generator model (pkl)')
    parser.add_argument('--out', default='./eval_out_drag', help='Output directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for generation')
    parser.add_argument('--num-points', type=int, default=2, 
                        help='Number of drag points (1-4)')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Number of drag optimization iterations')
    parser.add_argument('--mask-type', type=str, default='center_circle',
                        choices=['center_circle', 'center_rect', 'all_ones'],
                        help='Type of mask region')
    parser.add_argument('--cascaded-blending', action='store_true',
                        help='Enable cascaded blending after drag optimization')
    args = parser.parse_args()
    
    run_drag_eval(
        args.model,
        seed=args.seed,
        out_dir=args.out,
        num_drag_points=args.num_points,
        drag_iterations=args.iterations,
        mask_type=args.mask_type,
        use_cascaded_blending=args.cascaded_blending
    )