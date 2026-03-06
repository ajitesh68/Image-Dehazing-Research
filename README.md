# Image Dehazing Research Project

A PyTorch implementation of a U-Net based architecture for single image dehazing. This project includes custom dataset loaders, augmentations, and training pipelines optimized for the RESIDE-6K dataset.

## Structure
- `configs/`: YAML configuration files
- `data/`: Dataset loaders and augmentation pipelines
- `models/`: U-Net architecture, losses (MSE, L1, SSIM), and metrics (PSNR)
- `utils/`: Visualization and metrics tracking Utilities
- `train.py`: Main training loop script
- `evaluate.py`: Evaluation script on test datasets
- `inference.py`: Single image inference script

## Requirements
```bash
pip install torch torchvision pyyaml matplotlib pillow
```

## Dataset
This project is configured to use the RESIDE dataset. Download the dataset and place it according to the structure defined in `configs/default.yaml`.
