import sys
import os
import argparse
import glob
import numpy as np
import torch
from PIL import Image
import cv2

# Ensure we can import from local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from raft.raft import RAFT
    from raft.utils import flow_viz
    from raft.utils.utils import InputPadder
except ImportError:
    # Try appending 'raft' subdirectory explicitly if needed, though structure suggests raft package is in current dir
    sys.path.append(os.path.join(current_dir, 'raft'))
    from raft.raft import RAFT
    from raft.utils import flow_viz
    from raft.utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def save_flow_viz(img, flo, save_path):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # Map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    
    # Concatenate image and flow visualization
    # img is 0-255? Input to RAFT is usually 0-255.
    # flow_viz returns uint8 0-255
    
    img_flo = np.concatenate([img, flo], axis=1) # Side by side
    
    # Save image
    cv2.imwrite(save_path, img_flo[:, :, [2,1,0]].astype(np.uint8))

def run_raft_on_example(args):
    model_path = args.model
    if not os.path.exists(model_path):
        # Fallback to default path if argument not provided or incorrect
        default_path = os.path.join(current_dir, 'checkpoints', 'raft-things.pth')
        if os.path.exists(default_path):
            print(f"Model path {model_path} not found. Using default: {default_path}")
            model_path = default_path
        else:
            print(f"Error: Model not found at {model_path} or {default_path}")
            return

    # Initialize Model
    # Note: args needs to match what RAFT expects. 
    # RAFT class uses args.small, args.mixed_precision, args.alternate_corr
    
    model = torch.nn.DataParallel(RAFT(args))
    
    # Load checkpoint
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # Handle potential DataParallel wrapping in checkpoint if needed, 
    # but RAFT constructor typically expects clean state dict or we clean it.
    # Usually RAFT checkpoints are standard.
    # If the checkpoint keys start with 'module.', we might need to fix them, 
    # but torch.nn.DataParallel(RAFT(args)) wraps it, so loading into it expects 'module.' keys if saved that way.
    # Often standard RAFT checkpoints don't have 'module.'.
    
    new_state_dict = {}
    is_data_parallel_ckpt = any(k.startswith('module.') for k in state_dict.keys())
    
    if is_data_parallel_ckpt:
        model.load_state_dict(state_dict)
    else:
        # Checkpoint is simple, model is DataParallel. 
        # We can either load into model.module or use standard load_state_dict which might fail if keys mismatch.
        # Best to load into model.module
        model.module.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # Find images
    example_dir = os.path.join(current_dir, 'example')
    images = glob.glob(os.path.join(example_dir, '*.png'))
    
    if not images:
        print(f"No .png images found in {example_dir}")
        return

    # Sort numerically: 0.png, 1.png, ...
    # Filenames are like '0.png', '1.png'
    try:
        images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        print("Warning: Filenames in example/ are not all numeric. Sorting alphabetically.")
        images.sort()

    output_dir = os.path.join(current_dir, 'example_flow')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Found {len(images)} images. Processing flows to {output_dir}...")

    with torch.no_grad():
        for i, (imfile1, imfile2) in enumerate(zip(images[:-1], images[1:])):
            print(f"Processing {os.path.basename(imfile1)} -> {os.path.basename(imfile2)}")
            
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1_padded, image2_padded = padder.pad(image1, image2)

            flow_low, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
            
            # Unpad if necessary for visualization, though flow_viz handles flow.
            # flow_up is typically [1, 2, H, W]
            
            # If we want exact match to original image size:
            # flow_up = padder.unpad(flow_up)
            # image1 = image1 (original size)
            
            # Since flow_viz expects image and flow to match, let's unpad flow.
            flow_up = padder.unpad(flow_up)
            
            save_name = os.path.join(output_dir, f"flow_{i:04d}.png")
            save_flow_viz(image1, flow_up, save_name)

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint", default='checkpoints/raft-things.pth')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # If args.model relies on relative path, make it absolute or check existence
    if not os.path.isabs(args.model):
        args.model = os.path.join(current_dir, args.model)

    run_raft_on_example(args)
