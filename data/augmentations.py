import torch
from torchvision import transforms

def get_train_augmentations():
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.3)])

def get_val_augmentations():
    return None