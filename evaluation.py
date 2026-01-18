import os
import json
import numpy as np
from PIL import Image
import traceback

# ---- LPIPS deps (optional) ----
try:
    import torch
    import lpips
except Exception:
    torch = None
    lpips = None


def ensure_uint8_hwc(img):
    """Convert img to uint8 numpy array with shape (H, W, 3)."""
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        return arr

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        # CHW -> HWC
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        # drop alpha
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


def make_diff_visual(img1_u8, img2_u8, mask_f, brighten_delta=25):
    """
    生成差分可视化图：
      diff = abs(img2 - img1)
    并在 mask < 0.5 的区域对 diff 做轻微提亮（+brighten_delta）。

    Returns:
        diff_vis_u8: HxWx3 uint8
    """
    img1 = ensure_uint8_hwc(img1_u8)
    img2 = ensure_uint8_hwc(img2_u8)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: img1={img1.shape}, img2={img2.shape}")

    H, W = img1.shape[:2]
    if mask_f.shape != (H, W):
        raise ValueError(f"Mask shape mismatch: mask={mask_f.shape}, expected={(H, W)}")

    diff = np.abs(img2.astype(np.int16) - img1.astype(np.int16)).astype(np.uint8)

    bg = (mask_f < 0.5)
    if brighten_delta is not None and brighten_delta != 0:
        diff_i16 = diff.astype(np.int16)
        diff_i16[bg] = np.clip(diff_i16[bg] + int(brighten_delta), 0, 255)
        diff = diff_i16.astype(np.uint8)

    return diff


def masked_l1(img1_u8, img2_u8, mask_f, thr=0.5):
    """
    计算 masked L1：
      只在 mask > thr 的像素点上，计算 |img2-img1| 的平均值（RGB 三通道一起平均）。

    Returns:
        l1_mean: float
        num_pixels: int
    """
    img1 = ensure_uint8_hwc(img1_u8).astype(np.float32)
    img2 = ensure_uint8_hwc(img2_u8).astype(np.float32)

    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: img1={img1.shape}, img2={img2.shape}")

    H, W = img1.shape[:2]
    if mask_f.shape != (H, W):
        raise ValueError(f"Mask shape mismatch: mask={mask_f.shape}, expected={(H, W)}")

    m = (mask_f > thr)
    num = int(m.sum())
    if num == 0:
        return float("nan"), 0

    diff = np.abs(img2 - img1)  # HxWx3
    l1_mean = float(diff[m].mean())  # (num_pixels, 3) -> scalar mean

    return l1_mean, num


# -----------------------
# LPIPS helpers
# -----------------------
def _u8_to_lpips_tensor(img_u8: np.ndarray, device: str):
    """
    uint8 HWC -> torch tensor [1,3,H,W] in [-1,1]
    """
    if torch is None:
        raise RuntimeError("torch not available")
    x = torch.from_numpy(img_u8.astype(np.float32) / 255.0)  # HWC [0,1]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    x = x * 2.0 - 1.0  # [-1,1]
    return x.to(device)


def compute_lpips(img1_u8, img2_u8, net="alex", device="cuda"):
    """
    Compute LPIPS(image1, image2).
    Returns float or None if deps missing.
    """
    if lpips is None or torch is None:
        return None

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = lpips.LPIPS(net=net).to(device)
    model.eval()

    x = _u8_to_lpips_tensor(ensure_uint8_hwc(img1_u8), device)
    y = _u8_to_lpips_tensor(ensure_uint8_hwc(img2_u8), device)

    with torch.no_grad():
        d = model(x, y)
    return float(d.reshape(-1)[0].item())


class DragEvaluator:
    """
    Evaluator for drag evaluation results.

    输入：一个文件夹，包含三要素：
      - 初始图（image1*.png）
      - 处理后图（image2*.png 或 image2_blended*.png）
      - mask.png（灰度，0~255；会归一化到 0~1）

    输出：
      1) diff_vis.png：|img2-img1| 的差分图，并在 mask<0.5 区域轻微提亮
      2) masked L1：仅在 mask>0.5 的像素点上计算 L1
      3) LPIPS：原图 vs 生成图的感知距离（可选，若安装了 lpips/torch）
    """

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.img1 = None
        self.img2 = None
        self.mask = None
        self.points = None
        self.loaded = False

    def _find_file(self, prefix, suffix):
        """Find file in folder by prefix and suffix."""
        if not os.path.exists(self.folder_path):
            return None
        for fn in os.listdir(self.folder_path):
            if fn.startswith(prefix) and fn.endswith(suffix):
                return os.path.join(self.folder_path, fn)
        return None

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

        # Load image2 (prefer image2*.png, else try image2_blended*.png)
        img2_path = self._find_file('image2', '.png')
        if img2_path is None:
            img2_path = self._find_file('image2_blended', '.png')
            if img2_path is None:
                raise FileNotFoundError("Could not find image2_*.png or image2_blended*.png in folder")

        print(f"  Loading image2: {os.path.basename(img2_path)}")
        self.img2 = ensure_uint8_hwc(Image.open(img2_path))
        print(f"    Shape: {self.img2.shape}")

        # Load mask
        mask_path = self._find_file('mask', '.png')
        if mask_path is None:
            print("  WARNING: Could not find mask*.png, creating default mask (all ones)")
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
            with open(points_path, 'r', encoding='utf-8') as f:
                self.points = json.load(f)
            handle_points = self.points.get('handle_points', [])
            target_points = self.points.get('target_points', [])
            print(f"    Handle points: {len(handle_points)}, Target points: {len(target_points)}")
        else:
            print("  INFO: No drag_points.json or points.json found")
            self.points = None

        self.loaded = True
        print("Image loading complete.\n")

    def evaluate(
        self,
        brighten_delta=25,
        thr=0.5,
        save_diff_name="diff_vis.png",
        # LPIPS options
        compute_lpips_flag=True,
        lpips_net="alex",
        lpips_device="cuda",
    ):
        """
        Evaluate and write:
          - diff_vis.png
          - evaluation_results.json (with masked_l1 + lpips)

        Returns:
            results dict
        """
        if not self.loaded:
            raise RuntimeError("Images not loaded. Call load_images() first.")

        print("Starting evaluation...")
        print(f"  image1 shape: {self.img1.shape}")
        print(f"  image2 shape: {self.img2.shape}")
        print(f"  mask shape: {self.mask.shape}")

        if self.img1.shape != self.img2.shape:
            raise ValueError(f"Image shape mismatch: img1={self.img1.shape}, img2={self.img2.shape}")

        # 1) diff visualization
        diff_vis = make_diff_visual(self.img1, self.img2, self.mask, brighten_delta=brighten_delta)
        diff_path = os.path.join(self.folder_path, save_diff_name)
        Image.fromarray(diff_vis).save(diff_path)
        print(f"  Saved diff visualization to: {diff_path}")

        # 2) masked L1
        l1_mean, num_pix = masked_l1(self.img1, self.img2, self.mask, thr=thr)
        print(f"  Masked L1 (mask>{thr}): {l1_mean:.6f}  over pixels={num_pix}")

        # 3) LPIPS (optional)
        lpips_val = None
        if compute_lpips_flag:
            lpips_val = compute_lpips(self.img1, self.img2, net=lpips_net, device=lpips_device)
            if lpips_val is None:
                print("  [LPIPS] skipped (lpips/torch not available)")
            else:
                print(f"  [LPIPS] ({lpips_net}) = {lpips_val:.6f}")

        results = {
            'folder': self.folder_path,
            'image1_shape': list(self.img1.shape),
            'image2_shape': list(self.img2.shape),
            'mask_shape': list(self.mask.shape),
            'diff_vis_path': save_diff_name,
            'diff_brighten_delta_on_mask_lt_0p5': int(brighten_delta),
            'masked_l1_threshold': float(thr),
            'masked_l1_mean': float(l1_mean),
            'masked_l1_num_pixels': int(num_pix),
            'lpips': lpips_val,
            'lpips_net': lpips_net if lpips_val is not None else None,
        }

        if self.points is not None:
            results['num_points'] = int(len(self.points.get('handle_points', [])))

        print("Evaluation complete.\n")
        return results


def run_evaluation(folder_path, brighten_delta=25, thr=0.5):
    """Run evaluation on a single folder."""
    print(f"=== Drag Evaluation ===")
    print(f"Folder: {folder_path}\n")

    try:
        evaluator = DragEvaluator(folder_path)
        evaluator.load_images()
        results = evaluator.evaluate(
            brighten_delta=brighten_delta,
            thr=thr,
            compute_lpips_flag=True,   # 开启 LPIPS
            lpips_net="alex",
            lpips_device="cuda",
        )

        # Save results
        output_path = os.path.join(folder_path, 'evaluation_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")

        return results
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate drag results (diff vis + masked L1 + LPIPS)')
    parser.add_argument('--folder', required=True, help='Path to folder containing drag results')
    parser.add_argument('--brighten-delta', type=int, default=25,
                        help='Brightness delta added to diff image where mask < 0.5 (default: 25)')
    parser.add_argument('--thr', type=float, default=0.5,
                        help='Threshold for masked L1, compute over pixels where mask > thr (default: 0.5)')
    parser.add_argument('--no-lpips', action='store_true',
                        help='Disable LPIPS computation')
    parser.add_argument('--lpips-net', type=str, default='alex', choices=['alex', 'vgg', 'squeeze'],
                        help='LPIPS backbone network')
    parser.add_argument('--lpips-device', type=str, default='cuda',
                        help='Device for LPIPS (cuda or cpu)')
    args = parser.parse_args()

    evaluator = DragEvaluator(args.folder)
    evaluator.load_images()
    results = evaluator.evaluate(
        brighten_delta=args.brighten_delta,
        thr=args.thr,
        compute_lpips_flag=(not args.no_lpips),
        lpips_net=args.lpips_net,
        lpips_device=args.lpips_device,
    )

    output_path = os.path.join(args.folder, 'evaluation_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")
