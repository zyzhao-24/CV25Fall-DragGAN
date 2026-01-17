# Tracking Analysis

## 1. Points Data Type Analysis

The `points` variable in `core.py` is **NOT** stored as `uint8`.

*   **Logic**: `points` represents coordinates $[y, x]$ in the image.
*   **Storage**: In Python, it is a list of lists (e.g., `[[y1, x1], [y2, x2]]`). In PyTorch tensors (e.g., `renderer.points0_pt`), it is stored as `Float` (float32).
*   **Range**: The image resolution is typically $512 \times 512$ or larger. `uint8` can only represent values from 0 to 255. If `points` were `uint8`, it would not be able to represent coordinates greater than 255 (e.g., the right/bottom half of a $512 \times 512$ image), nor could it represent sub-pixel precision (decimals).
*   **Confusion Clarification**: The code does convert the image to the 0-255 range (`curr_img_raft = (img * 127.5 + 128).clamp(0, 255)`), but the tensor data type remains `float32` (unless explicitly cast to `byte/uint8`), and this applies to pixel *colors*, not the coordinate *points*.

## 2. L2 Tracking Implementation

In `core.py`, the L2 tracking process is essentially: **Feature Extraction -> Bilinear Upsampling -> Nearest Neighbor Search**.

The specific steps are:

1.  **Feature Extraction**: The generator $G$ produces low-resolution features `feat[feature_idx]`.
2.  **Upsampling (Critical Step)**:
    ```python
    feat_resize = F.interpolate(feat[feature_idx], [h, w], mode='bilinear')
    ```
    The code explicitly uses `bilinear` interpolation to upsample the feature map to the original image resolution ($H, W$).
3.  **Search**: `point_tracking_L2_point` functions on this high-resolution `feat_resize` map. It searches for the pixel location in a local window that has the minimum L2 distance to the reference feature.

## 3. RAFT Tracking Analysis

You asked: *"So should RAFT also be upsampled?"*

**Answer**: Theoretically, RAFT outputs **already include upsampling**, so no manual external upsampling is required in `core.py`, assuming the utilized RAFT model behaves as per standard.

*   **Input**: RAFT takes the RGB images ($H, W$) as input.
*   **Internal Processing**: RAFT internally computes flow at a lower resolution (usually $1/8$ of the input).
*   **Output**: The standard RAFT model (`RAFT` class) returns two flows: `flow_low` and `flow_up`.
    ```python
    flow_low, flow_up = renderer.raft_model(image1, image2, iters=20, test_mode=True)
    ```
*   **Upsampling Mechanism**: `flow_up` is the optical flow that has already been upsampled to the original input resolution ($H, W$) by the RAFT network (often using a learned convex upsampling mask or bilinear interpolation inside the model's `forward` pass).
*   **Conclusion**: Since `core.py` (Line 158) uses `flow_up`, the flow field is already at the correct resolution ($H, W$). Unlike L2 tracking where we manually upsample the feature map, for RAFT, the model handles the resolution matching.
