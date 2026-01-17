import os
import sys
import json
import numpy as np
from PIL import Image
import traceback

def ensure_uint8_hwc(img):
    """Convert img to uint8 numpy array with shape (H, W, 3)."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr
    
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
    
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            vmin, vmax = float(arr.min()), float(arr.max())
            if vmax <= 1.0 and vmin >= 0.0:
                arr = (arr * 255.0).round()
            elif vmax <= 1.0 and vmin >= -1.0:
                arr = ((arr * 0.5 + 0.5) * 255.0).round()
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Image has invalid shape: {arr.shape}, dtype={arr.dtype}")
    
    return arr


class DragEvaluator:
    """Evaluator for drag evaluation results."""
    
    def __init__(self, folder_path):
        """
        Initialize evaluator with a folder containing drag results.
        
        Args:
            folder_path: Path to folder containing:
                - image1_original.png (or similar)
                - image2_dragged.png or image2_dragged_blended.png
                - mask.png
                - drag_points.json (optional)
        """
        self.folder_path = folder_path
        self.img1 = None
        self.img2 = None
        self.mask = None
        self.points = None
        self.loaded = False
    
    def load_images(self):
        """Load image1, image2, and mask from folder."""
        print(f"Loading images from: {self.folder_path}")
        
        # Load image1
        img1_path = self._find_file('image1', '.png')
        if img1_path is None:
            raise FileNotFoundError("Could not find image1_*.png in folder")
        
        print(f"  Loading image1: {os.path.basename(img1_path)}")
        self.img1 = ensure_uint8_hwc(Image.open(img1_path))
        print(f"    Shape: {self.img1.shape}")
        
        # Load image2
        img2_path = self._find_file('image2', '.png')
        if img2_path is None:
            img2_path = self._find_file('image2_blended','.png')
            if img2_path is None:
                raise FileNotFoundError("Could not find image2_*.png in folder")
        
        print(f"  Loading image2: {os.path.basename(img2_path)}")
        self.img2 = ensure_uint8_hwc(Image.open(img2_path))
        print(f"    Shape: {self.img2.shape}")
        
        # Load mask
        mask_path = self._find_file('mask', '.png')
        if mask_path is None:
            print("  WARNING: Could not find mask_*.png, creating default mask (all ones)")
            self.mask = np.ones(self.img1.shape[:2], dtype=np.float32)
        else:
            print(f"  Loading mask: {os.path.basename(mask_path)}")
            mask_img = Image.open(mask_path).convert('L')
            self.mask = np.array(mask_img, dtype=np.float32) / 255.0
            print(f"    Shape: {self.mask.shape}, Range: [{self.mask.min():.3f}, {self.mask.max():.3f}]")
        
        # Load points (optional)
        points_path = self._find_file('drag_points', '.json')
        if points_path is None:
            points_path = self._find_file('points', '.json')
        
        if points_path is not None:
            print(f"  Loading points: {os.path.basename(points_path)}")
            with open(points_path, 'r') as f:
                self.points = json.load(f)
            handle_points = self.points.get('handle_points', [])
            target_points = self.points.get('target_points', [])
            print(f"    Handle points: {len(handle_points)}, Target points: {len(target_points)}")
        else:
            print("  INFO: No drag_points.json or points.json found")
            self.points = None
        
        self.loaded = True
        print("Image loading complete.\n")
    
    def _find_file(self, prefix, suffix):
        """Find file in folder by prefix and suffix."""
        if not os.path.exists(self.folder_path):
            return None
        
        for fn in os.listdir(self.folder_path):
            if fn.startswith(prefix) and fn.endswith(suffix):
                return os.path.join(self.folder_path, fn)
        return None
    
    def get_image1(self):
        """Get original image."""
        if not self.loaded:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        return self.img1
    
    def get_image2(self):
        """Get generated/dragged image."""
        if not self.loaded:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        return self.img2
    
    def get_mask(self):
        """Get mask."""
        if not self.loaded:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        return self.mask
    
    def get_points(self):
        """Get drag points."""
        if not self.loaded:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        return self.points
    
    def evaluate(self):
        """
        Evaluate the drag results.
        
        Returns:
            Dictionary with evaluation results.
        """
        if not self.loaded:
            raise RuntimeError("Images not loaded. Call load_images() first.")
        
        print("Starting evaluation...")
        print(f"  image1 shape: {self.img1.shape}")
        print(f"  image2 shape: {self.img2.shape}")
        print(f"  mask shape: {self.mask.shape}")
        
        results = {
            'folder': self.folder_path,
            'image1_shape': self.img1.shape,
            'image2_shape': self.img2.shape,
            'mask_shape': self.mask.shape,
            # Add more metrics here as needed
        }
        
        if self.points is not None:
            results['num_points'] = len(self.points.get('handle_points', []))
        
        print("Evaluation complete.\n")
        return results


def run_evaluation(folder_path):
    """Run evaluation on a single folder or batch of folders."""
    print(f"=== Drag Evaluation ===")
    print(f"Folder: {folder_path}\n")
    
    try:
        evaluator = DragEvaluator(folder_path)
        evaluator.load_images()
        results = evaluator.evaluate()
        
        # Save results
        output_path = os.path.join(folder_path, 'evaluation_results.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
        
        return results
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate drag results')
    parser.add_argument('--folder', required=True, help='Path to folder containing drag results')
    args = parser.parse_args()
    
    run_evaluation(args.folder)
