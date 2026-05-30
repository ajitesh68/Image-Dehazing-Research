import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def tensor_to_numpy(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]

    img = tensor.cpu().detach().permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)

    return img


def save_comparison(hazy_batch, predicted_batch, clean_batch, save_path,
                    num_images=4):
    num_images = min(num_images, hazy_batch.size(0))

    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        axes[i, 0].imshow(tensor_to_numpy(hazy_batch[i]))
        axes[i, 0].set_title('Hazy Input', fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(tensor_to_numpy(predicted_batch[i]))
        axes[i, 1].set_title('Dehazed (Ours)', fontsize=12)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(tensor_to_numpy(clean_batch[i]))
        axes[i, 2].set_title('Ground Truth', fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📸 Comparison image save ho gayi: {save_path}")


def plot_training_curves(train_losses, val_losses, train_psnrs=None,
                         val_psnrs=None, save_path=None):
    fig, axes = plt.subplots(1, 2 if train_psnrs else 1,
                             figsize=(14 if train_psnrs else 7, 5))

    if train_psnrs:
        ax1, ax2 = axes
    else:
        ax1 = axes

    ax1.plot(train_losses, label='Train Loss', color='#2196F3', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', color='#F44336', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    if train_psnrs:
        ax2.plot(train_psnrs, label='Train PSNR', color='#4CAF50', linewidth=2)
        ax2.plot(val_psnrs, label='Val PSNR', color='#FF9800', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('PSNR (dB)', fontsize=12)
        ax2.set_title('Training & Validation PSNR', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Training curves save ho gayi: {save_path}")
    else:
        plt.show()