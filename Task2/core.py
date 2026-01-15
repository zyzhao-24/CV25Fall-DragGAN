import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'raft'))

from raft.utils import flow_viz
from raft.utils.utils import InputPadder
from raft.raft import RAFT
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
import copy

# Import cascaded blending module
from core_blending import apply_cascaded_blending, prepare_blending_mask


# Handle tracking method 'raft', 'L2'
tracking_method = 'L2'
is_save = False


class RAFTArgs:
    def __init__(self, model='', small=False, mixed_precision=False, alternate_corr=False):
        self.model = model
        self.small = small
        self.dropout = 0
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr

    def __contains__(self, key):
        return hasattr(self, key)


def point_tracking_L2_point(renderer, feat_resize, points, r2, h, w):
    with torch.no_grad():
        for j, point in enumerate(points):
            r = round(r2 / 512 * h)
            up = max(point[0] - r, 0)
            down = min(point[0] + r + 1, h)
            left = max(point[1] - r, 0)
            right = min(point[1] + r + 1, w)
            feat_patch = feat_resize[:, :, up:down, left:right]
            L2 = torch.linalg.norm(
                feat_patch - renderer.feat_refs[j].reshape(1, -1, 1, 1), dim=1)
            _, idx = torch.min(L2.view(1, -1), -1)
            width = right - left
            point = [idx.item() // width + up, idx.item() % width + left]
            points[j] = point
    return points


def save_flow_viz(img, flo, save_path):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # Map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    # Concatenate image and flow visualization
    img_flo = np.concatenate([img, flo], axis=1)  # Side by side

    # Save image
    cv2.imwrite(save_path, img_flo[:, :, [2, 1, 0]].astype(np.uint8))


def point_tracking_raft(renderer, points, last_img, curr_img, h, w):
    if not hasattr(renderer, 'raft_model') or renderer.raft_model is None:
        print("Initializing RAFT model...")
        ckpt = 'checkpoints/raft-things.pth'
        args = RAFTArgs(model=ckpt)

        if not os.path.exists(ckpt):
            print(
                f"Warning: RAFT checkpoint not found at {ckpt}. RAFT tracking will fail.")
            return points

        renderer.raft_model = torch.nn.DataParallel(RAFT(args))
        renderer.raft_model.load_state_dict(torch.load(ckpt))
        renderer.raft_model = renderer.raft_model.module
        renderer.raft_model.to(renderer._device)
        renderer.raft_model.eval()

    with torch.no_grad():
        # Prepare images: [1, 3, H, W], 0-255
        # Ensure inputs are tensor, on device, etc.
        # last_img and curr_img are expected to be [1, 3, H, W]

        padder = InputPadder(last_img.shape)
        image1, image2 = padder.pad(last_img, curr_img)

        # RAFT forward pass
        # image1, image2 are 0-255
        flow_low, flow_up = renderer.raft_model(
            image1, image2, iters=20, test_mode=True)

        # Unpad flow to match original image dimensions
        flow_up = padder.unpad(flow_up)

        if is_save:
            save_dir = 'example_flow'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_count = getattr(renderer, 'save_count', 0)
            save_path = os.path.join(save_dir, f'flow_{save_count:04d}.png')
            save_flow_viz(last_img, flow_up, save_path)
        
        new_points = []
        # Update points
        for j, point in enumerate(points):
            # Point is [y, x]
            py, px = point[0], point[1]

            # Simple integer access (nearest neighbor)
            iy, ix = int(round(py)), int(round(px))
            iy = min(max(iy, 0), h-1)
            ix = min(max(ix, 0), w-1)

            # flow_up channel 0 is x-displacement, channel 1 is y-displacement
            # Wait, verify RAFT output channels
            # RAFT usually outputs [delta_x, delta_y] at dim=1
            dx = flow_up[0, 0, iy, ix].item()
            dy = flow_up[0, 1, iy, ix].item()

            point = [py + dy, px + dx]
            points[j] = point

    return points


def motion_supervision(renderer, feat_resize, points, targets, r1, h, w):
    loss_motion = 0
    stop = True
    X = torch.linspace(0, h, h)
    Y = torch.linspace(0, w, w)
    xx, yy = torch.meshgrid(X, Y)

    for j, point in enumerate(points):
        direction = torch.Tensor(
            [targets[j][1] - point[1], targets[j][0] - point[0]])
        if torch.linalg.norm(direction) > max(2 / 512 * h, 2):
            stop = False
        if torch.linalg.norm(direction) > 1:
            distance = ((xx.to(renderer._device) -
                        point[0])**2 + (yy.to(renderer._device) - point[1])**2)**0.5
            relis, reljs = torch.where(distance < round(r1 / 512 * h))
            direction = direction / (torch.linalg.norm(direction) + 1e-7)
            gridh = (relis+direction[1]) / (h-1) * 2 - 1
            gridw = (reljs+direction[0]) / (w-1) * 2 - 1
            grid = torch.stack([gridw, gridh], dim=-
                               1).unsqueeze(0).unsqueeze(0)
            target = F.grid_sample(feat_resize.float(),
                                   grid, align_corners=True).squeeze(2)
            loss_motion += F.l1_loss(feat_resize[:,
                                     :, relis, reljs].detach(), target)

    return loss_motion, stop


def render_drag_impl(renderer, res,
                     points=[],
                     targets=[],
                     mask=None,
                     lambda_mask=10,
                     reg=0,
                     feature_idx=5,
                     r1=3,
                     r2=12,
                     random_seed=0,
                     noise_mode='const',
                     trunc_psi=0.7,
                     force_fp32=False,
                     layer_name=None,
                     sel_channels=3,
                     base_channel=0,
                     img_scale_db=0,
                     img_normalize=False,
                     untransform=False,
                     is_drag=False,
                     reset=False,
                     to_pil=False,
                     **kwargs
                     ):
    G = renderer.G
    ws = renderer.w
    if ws.dim() == 2:
        ws = ws.unsqueeze(1).repeat(1, 6, 1)
    ws = torch.cat([ws[:, :6, :], renderer.w0[:, 6:, :]], dim=1)
    if hasattr(renderer, 'points'):
        if len(points) != len(renderer.points):
            reset = True
    if reset:
        renderer.feat_refs = None
        renderer.points0_pt = None
        renderer.last_image = None
        renderer.save_count = 0
        if not os.path.exists('example'):
            os.makedirs('example')
        with open('example/points_log.txt', 'w') as f:
            f.write('')
    renderer.points = points

    # Run synthesis network.
    label = torch.zeros([1, G.c_dim], device=renderer._device)
    img, feat = G(ws, label, truncation_psi=trunc_psi,
                  noise_mode=noise_mode, input_is_w=True, return_feature=True)

    h, w = G.img_resolution, G.img_resolution

    # Prepare current image for RAFT (0-255 Tensor)
    # img from G is approx [-1, 1], shape [1, 3, H, W]
    curr_img_raft = (img * 127.5 + 128).clamp(0, 255)

    if is_drag:
        feat_resize = F.interpolate(feat[feature_idx], [h, w], mode='bilinear')
        if renderer.feat_refs is None:
            renderer.feat0_resize = F.interpolate(
                feat[feature_idx].detach(), [h, w], mode='bilinear')
            renderer.feat_refs = []
            for point in points:
                py, px = round(point[0]), round(point[1])
                renderer.feat_refs.append(renderer.feat0_resize[:, :, py, px])
            renderer.points0_pt = torch.Tensor(points).unsqueeze(
                0).to(renderer._device)  # 1, N, 2

        if tracking_method == 'raft':
            if hasattr(renderer, 'last_image') and renderer.last_image is not None:
                points = point_tracking_raft(
                    renderer, points, renderer.last_image, curr_img_raft, h, w)
            else:
                points = points
            # L2_points = point_tracking_L2_point(renderer, feat_resize, points, r2, h, w)
            
            # sum_point_distance = 0

            # for j in range(len(points)):
            #     p1 = np.array(points[j], dtype=float)
            #     p2 = np.array(L2_points[j], dtype=float)
            #     sum_point_distance += np.linalg.norm(p1 - p2)

            # print(f'Distance between two points tracking methods{sum_point_distance}')
            
        elif tracking_method == 'L2':
            # Default Point tracking with feature matching
            points = point_tracking_L2_point(
                renderer, feat_resize, points, r2, h, w)
        else:
            print('No matching point tracking method')

        res.points = [[point[0], point[1]] for point in points]

        # Motion supervision
        loss_motion, res.stop = motion_supervision(
            renderer, feat_resize, points, targets, r1, h, w)

        loss = loss_motion
        if mask is not None:
            if mask.min() == 0 and mask.max() == 1:
                mask_usq = mask.to(renderer._device).unsqueeze(0).unsqueeze(0)
                loss_fix = F.l1_loss(feat_resize * mask_usq,
                                     renderer.feat0_resize * mask_usq)
                loss += lambda_mask * loss_fix

        loss += reg * F.l1_loss(ws, renderer.w0)  # latent code regularization
        if not res.stop:
            renderer.w_optim.zero_grad()
            loss.backward()
            renderer.w_optim.step()

    # Update last_image for next iteration
    # renderer.last_image = curr_img_raft.detach()

    # Scale and convert to uint8.
    img = img[0]
    if img_normalize:
        img = img / img.norm(float('inf'),
                             dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
    img = img * (10 ** (img_scale_db / 20))
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)

    # Check if cascaded blending is needed
    if (hasattr(renderer, 'mask_snapshot_features') and 
        renderer.mask_snapshot_features is not None and
        mask is not None and mask.min() == 0 and mask.max() == 1):
        
        try:
            # Prepare blending mask
            blending_mask = prepare_blending_mask(mask, renderer._device)
            
            # Calculate blending coefficients: start from layer 6, alpha=0.5 for first layer, then halve each subsequent layer
            snapshot_features = renderer.mask_snapshot_features
            num_layers = len(snapshot_features)
            start_idx = 6  # Start blending from layer 6 to avoid affecting gradient-computed layers
            
            # Create blending coefficients list
            blend_coeffs = [0.0] * start_idx  # Layers before start_idx don't blend
            
            # Calculate blending coefficients: alpha=0.5 for first blending layer, then halve original feature weight
            for i in range(start_idx, num_layers):
                layer_idx = i - start_idx  # Index in blending layers (0 for first blending layer)
                # alpha is the weight for snapshot features: 0.5, 0.25, 0.125, ...
                alpha = 0.5 ** (layer_idx + 1)
                blend_coeffs.append(alpha)
            
            print(f"[core] Calculating blending coefficients: starting from layer {start_idx}")
            print(f"[core] Blending coefficients example: {blend_coeffs[start_idx:start_idx+min(5, len(blend_coeffs)-start_idx)]}...")
            
            # Generate cascaded blended image
            with torch.no_grad():
                blended_img, _ = apply_cascaded_blending(
                    G, ws, 
                    snapshot_features, 
                    blending_mask,
                    label=label,
                    truncation_psi=trunc_psi,
                    noise_mode=noise_mode,
                    start_idx=start_idx,
                    blend_coeffs=blend_coeffs
                )
            
            # Process blended image
            blended_img = blended_img[0]
            if img_normalize:
                blended_img = blended_img / blended_img.norm(float('inf'),
                                                           dim=[1, 2], keepdim=True).clip(1e-8, 1e8)
            blended_img = blended_img * (10 ** (img_scale_db / 20))
            blended_img = (blended_img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
            
            # Use blended image
            img = blended_img
            print(f"[core] Using cascaded blended image (mask region remains unchanged)")
        except Exception as e:
            print(f"[core] Cascaded blending failed: {e}")
            print(f"[core] Using original image")

    if is_save:
        if not hasattr(renderer, 'save_count'):
            renderer.save_count = 0
            if not os.path.exists('example'):
                os.makedirs('example')
            # Ensure log exists if we didn't reset
            with open('example/points_log.txt', 'w') as f:
                f.write('')

        try:
            Image.fromarray(img.cpu().numpy()).save(
                f'example/{renderer.save_count}.png')
            with open('example/points_log.txt', 'a') as f:
                f.write(f'{renderer.save_count}: {points}\n')
            renderer.save_count += 1
        except Exception as e:
            print(f"Error saving debug info: {e}")

    if to_pil:
        img = img.cpu().numpy()
        img = Image.fromarray(img)
    res.image = img
    res.w = ws.detach().cpu().numpy()
