# CV25Fall-DragGAN: Enhanced DragGAN with Advanced Tracking and Blending

This repository is part of submitted files of computer vision project for 2025 fall semester. It contains an enhanced implementation of DragGAN with advanced tracking methods and cascaded blending techniques.

**Team Members:**

- 2400017711 赵泽宇
- 2400017808 王唐欣宇  
- 2400017766 鄢宇阳

## Overview

This project extends the original DragGAN implementation with several key enhancements:

1. **Advanced Point Tracking Methods:**
   - **RAFT-based tracking**: Uses optical flow for smooth and accurate point tracking
   - **L2 feature matching**: Traditional feature-based tracking
   - **Mixed tracking**: Combines RAFT for coarse tracking with L2 for fine adjustment
   - **Area-wise tracking**: Adaptively chooses tracking method based on texture richness

2. **Enhanced Mask Preservation:**
   - **Fix Loss Options**: Provided 3 possible approch to improve original fix loss
   - **Cascaded blending**: Gradually blends features to maintain mask integrity

## 1. Installation

Install [Conda](https://www.anaconda.com/) and create a Conda environment using the provided environment.yml file, then manually install pytorch (due to requirement of cuda compiled package):

```bash
conda env create -f environment.yml
conda activate draggan_project
pip install torch==2.7.0+cu128 torchvision --extra-index-url https://download.pytorch.org/whl/cu128
```

### Download Pre-trained Models

Download the required pre-trained models:

```bash
python scripts/download_model.py
```

This will download StyleGAN2 and RAFT checkpoints to the `checkpoints` directory.

## 2. Quick Start with Web Interface

1. Launch the Gradio web interface:

    ```bash
    python visualizer.py
    ```

2. Open your browser and navigate to `http://localhost:7860`

3. **Basic Usage:**
   - Select a pretrained model from the dropdown
   - Adjust the seed to generate different initial images
   - Click on the image to add control points (start points)
   - Click again to set target points
   - Click "Start" to begin the drag operation
   - Click "Stop" to pause the operation

## 3. Advanced Features

### 3.1 Tracking Methods

The system supports four tracking methods:

1. **L2**: Traditional feature matching using L2 distance
2. **RAFT**: Optical flow-based tracking using RAFT model
3. **Mixed**: Combines RAFT for coarse tracking and L2 for fine adjustment
4. **Area-wise**: Adaptively selects tracking method based on texture richness

### 3.2 Fix Loss Types

Four types of mask preservation losses are available:

1. **Single**: Uses original loss
2. **Blended**: Uses cascaded blended features for L2 computation
3. **Multilayer**: Use L2 loss from multiple features from different layers
4. **RAFT**: Uses optical flow magnitude for fix loss

### 3.3 Cascaded Blending

When "Use blended output" is enabled, the system applies cascaded blending:

- Gradually blends features from different network layers
- Maintains mask region integrity while allowing deformation
- Provides smoother transitions in masked regions

## 4. Command Line Tools

### 4.1 Batch Evaluation

Run batch evaluation with different configurations:

```bash
# Original loss
bash scripts/eval_single.sh

# Multi-layer feature loss  
bash scripts/eval_multilayer.sh

# RAFT-based loss
bash scripts/eval_raft.sh

# Blend loss
bash scripts/eval_blend.sh

# Cascaded blending
bash scripts/eval_cascaded.sh
```

if you are using a windows system, directly paste the content of corresponding file into your terminal.

### 4.2 Tracking Method Evaluation

Evaluate different tracking methods:

```bash
# Mixed tracking
bash scripts/eval_tracking_mixed.sh

# RAFT tracking
bash scripts/eval_tracking_raft.sh

# Area-wise tracking
bash scripts/eval_tracking_areawise.sh
```

if you are using a windows system, directly paste the content of corresponding file into your terminal.

### 4.3 Generate variance heatmap

```bash
python variance_heatmap.py
```

## 5. Project Structure

```plain
CV25Fall-DragGAN/
├── core.py                    # Main drag implementation with tracking methods
├── core_blending.py           # Cascaded blending implementation
├── visualizer.py              # Gradio web interface
├── drag_auto.py               # Automated drag operations
├── drag_multi.py              # Multi-point drag operations
├── evaluation.py              # Evaluation utilities
├── variance_heatmap.py        # Visualize variance heatmap
├── environment.yml            # Conda environment configuration
├── scripts/                   # Evaluation and download scripts
├── checkpoints/               # Pre-trained model checkpoints
├── raft/                      # RAFT optical flow implementation
├── viz/                       # Visualization utilities
├── gradio_utils/              # Gradio UI utilities
├── torch_utils/               # PyTorch utilities
├── training/                  # Training code
└── dnnlib/                    # StyleGAN2 utilities
```

## Acknowledgments

- This project is based on the original [DragGAN](https://github.com/XingangPan/DragGAN) implementation
- Uses [RAFT](https://github.com/princeton-vl/RAFT) for optical flow computation
- Built with [PyTorch](https://pytorch.org/) and [Gradio](https://www.gradio.app/)

This project is only for homework purpose. Anyone using this repository must follow the requirements of original DragGAN and RAFT repository.
