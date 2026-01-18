import os
import sys
import torch
import dnnlib
import legacy
import numpy as np
import PIL.Image
import torch.nn.functional as F
import cv2

# Add current directory to path
sys.path.append(os.getcwd())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_local_std(image, r=5):
    """
    Compute local standard deviation for every pixel/area.
    image: [1, 1, H, W] float 0-255
    r: radius
    """
    k = 2 * r + 1
    padding = r
    
    # Calculate Mean E[X]
    # Use average pooling to compute sliding window mean
    mean_x = F.avg_pool2d(image, kernel_size=k, stride=1, padding=padding)
    
    # Calculate Mean E[X^2]
    mean_x2 = F.avg_pool2d(image**2, kernel_size=k, stride=1, padding=padding)
    
    # Var(X) = E[X^2] - (E[X])^2
    var = mean_x2 - mean_x**2
    
    # Numerical stability
    var = torch.clamp(var, min=0)
    return torch.sqrt(var)

def create_colorbar(height, width, max_val, colormap=cv2.COLORMAP_JET):
    # Gradient from 255 (top) to 0 (bottom)
    # 0 -> Blue (Low variance), 255 -> Red (High variance) in JET
    gradient = np.linspace(255, 0, height).astype(np.uint8)
    gradient = np.tile(gradient[:, None], (1, width))
    
    colorbar_img = cv2.applyColorMap(gradient, colormap)
    colorbar_img = cv2.cvtColor(colorbar_img, cv2.COLOR_BGR2RGB)
    
    # Add text labels
    # We need a canvas slightly wider for text
    canvas = np.ones((height, width + 60, 3), dtype=np.uint8) * 255 # White background
    canvas[:, :width] = colorbar_img
    
    # Draw ticks/labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)
    thickness = 1
    
    steps = 6 # 0, 10, 20, 30, 40, 50
    for i in range(steps):
        val = (i / (steps - 1)) * max_val
        # y position: val=0 at bottom (height), val=max at top (0)
        y = int(height - (i / (steps - 1)) * (height - 1))
        
        # Adjust y slightly to center text vertically on the tick
        y_text = y + 5
        if y_text > height - 5: y_text = height - 5
        if y_text < 15: y_text = 15
        
        cv2.putText(canvas, f"{val:.1f}", (width + 5, y_text), font, font_scale, color, thickness)
        
    return canvas

def generate_images(network_pkl, out_dir, model_name, num_imgs=10, seed_start=0):
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    
    os.makedirs(out_dir, exist_ok=True)
    
    for i in range(num_imgs):
        seed = seed_start + i
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        label = torch.zeros([1, G.c_dim], device=device)
        
        # Generate image
        # G returns [1, 3, H, W] in range roughly [-1, 1]
        img = G(z, label, truncation_psi=0.7, noise_mode='const')
        
        # Prepare RGB image for display (0-255)
        # Permute to H,W,C for saving
        img_vis = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_vis_np = img_vis[0].cpu().numpy() # [H, W, 3] RGB
        
        # Prepare grayscale for std computation matching core.py
        # core.py logic: img_gray = last_img.float().mean(dim=1, keepdim=True) where last_img was 0-255
        img_float = (img * 127.5 + 128).clamp(0, 255)
        img_gray = img_float.mean(dim=1, keepdim=True) # [1, 1, H, W]
        
        # Compute std
        std_map = compute_local_std(img_gray, r=5) 
        std_map_np = std_map[0, 0].cpu().numpy()
        
        # print(f"[{model_name} {i}] Mean local std: {np.mean(std_map_np):.2f}, Max: {np.max(std_map_np):.2f}")

        # Create heatmap
        # Visualize 0-50 range. The threshold is around 20.
        norm_std = np.clip(std_map_np / 50.0 * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap (JET goes Blue -> Green -> Red)
        # Blue (low var/textureless) -> Red (high var/texture rich)
        heatmap_img = cv2.applyColorMap(norm_std, cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB) # OpenCV is BGR
        
        # Create colorbar
        bar_width = 30
        colorbar = create_colorbar(img_vis_np.shape[0], bar_width, 50.0)
        
        # Concatenate side by side: Original | Heatmap | Colorbar
        combined = np.concatenate([img_vis_np, heatmap_img, colorbar], axis=1)
        
        # Save
        out_path = os.path.join(out_dir, f'{model_name}_{i:02d}.png')
        PIL.Image.fromarray(combined).save(out_path)
        print(f'Saved {out_path}')

if __name__ == "__main__":
    out_dir = "docs/variance_visualization"
    
    # Models
    models = [
        ('ffhq', 'checkpoints/stylegan2-ffhq-512x512.pkl'),
        ('car', 'checkpoints/stylegan2-car-config-f.pkl')
    ]
    
    for name, path in models:
        if not os.path.exists(path):
            print(f"Error: Checkpoint {path} not found.")
            continue
        
        try:
            generate_images(path, out_dir, name, num_imgs=10)
        except Exception as e:
            print(f"Failed to generate for {name}: {e}")
            import traceback
            traceback.print_exc()
