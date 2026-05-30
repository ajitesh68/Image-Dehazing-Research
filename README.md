# 🌫️ Image Dehazing Project

## Project Structure
```
image-dehazing-public/
├── configs/          # Hyperparameter configuration files
├── data/             # Data loading and augmentation code
├── models/           # Neural network architectures and losses
├── utils/            # Helper functions (visualization, logging)
├── experiments/      # Saved models and training logs
├── train.py          # Training script
├── evaluate.py       # Evaluation script
└── inference.py      # Run on single image
```

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --config configs/default.yaml
```

## Evaluation
```bash
python evaluate.py --checkpoint experiments/best_model.pth
```

## Inference
```bash
python inference.py --image path/to/foggy.jpg --output clear.jpg
```
