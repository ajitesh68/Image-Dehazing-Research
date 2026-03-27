"""
Generate Output for Supervisor — REAL data + GPU support!
Colab pe run karo: !python generate_demo_output.py

Output: practice/results/ folder mein 5 PNG images
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.unet import UNet


# ============= GPU AUTO-DETECT =============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", end="")
if device.type == 'cuda':
    print(f" ({torch.cuda.get_device_name(0)})")
else:
    print(" (GPU nahi mila — CPU pe chalega, thoda slow hoga)")


class RealDehazingDataset(Dataset):
    """RESIDE-6K ya koi bhi hazy-clean paired dataset load karo"""

    def __init__(self, hazy_dir, clean_dir, image_size=128, max_images=None):
        self.image_size = image_size
        self.pairs = []

        hazy_files = sorted(glob.glob(os.path.join(hazy_dir, '*.*')))
        clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.*')))

        # Match by filename
        clean_dict = {os.path.basename(f): f for f in clean_files}
        for hf in hazy_files:
            name = os.path.basename(hf)
            if name in clean_dict:
                self.pairs.append((hf, clean_dict[name]))

        if max_images and len(self.pairs) > max_images:
            self.pairs = self.pairs[:max_images]

        print(f"  Loaded {len(self.pairs)} paired images from {hazy_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, clean_path = self.pairs[idx]
        hazy = Image.open(hazy_path).convert('RGB').resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        clean = Image.open(clean_path).convert('RGB').resize(
            (self.image_size, self.image_size), Image.BILINEAR)

        hazy = torch.tensor(np.array(hazy).transpose(2, 0, 1).astype(np.float32) / 255.0)
        clean = torch.tensor(np.array(clean).transpose(2, 0, 1).astype(np.float32) / 255.0)
        return hazy, clean


def find_dataset():
    """Dataset automatically dhundho — multiple paths try karo"""
    possible_paths = [
        # Colab + kaggle download
        ('RESIDE-6K/RESIDE-6K/train/hazy', 'RESIDE-6K/RESIDE-6K/train/GT'),
        ('RESIDE-6K/train/hazy', 'RESIDE-6K/train/GT'),
        ('../RESIDE-6K/RESIDE-6K/train/hazy', '../RESIDE-6K/RESIDE-6K/train/GT'),
        ('../RESIDE-6K/train/hazy', '../RESIDE-6K/train/GT'),
        ('data/train/hazy', 'data/train/GT'),
        ('/content/RESIDE-6K/RESIDE-6K/train/hazy', '/content/RESIDE-6K/RESIDE-6K/train/GT'),
    ]

    test_paths = [
        ('RESIDE-6K/RESIDE-6K/test/hazy', 'RESIDE-6K/RESIDE-6K/test/GT'),
        ('RESIDE-6K/test/hazy', 'RESIDE-6K/test/GT'),
        ('../RESIDE-6K/RESIDE-6K/test/hazy', '../RESIDE-6K/RESIDE-6K/test/GT'),
        ('../RESIDE-6K/test/hazy', '../RESIDE-6K/test/GT'),
        ('/content/RESIDE-6K/RESIDE-6K/test/hazy', '/content/RESIDE-6K/RESIDE-6K/test/GT'),
    ]

    train_hazy, train_clean = None, None
    test_hazy, test_clean = None, None

    for h, c in possible_paths:
        if os.path.exists(h) and os.path.exists(c):
            train_hazy, train_clean = h, c
            break

    for h, c in test_paths:
        if os.path.exists(h) and os.path.exists(c):
            test_hazy, test_clean = h, c
            break

    return train_hazy, train_clean, test_hazy, test_clean


def download_dataset():
    """Colab pe RESIDE-6K download karo (agar available nahi hai)"""
    print("\nDataset dhundh rahe hain...")
    train_h, train_c, test_h, test_c = find_dataset()

    if train_h is None:
        print("Dataset nahi mila! Kaggle se download kar rahe hain...")
        try:
            os.system('pip install kagglehub -q')
            import kagglehub
            path = kagglehub.dataset_download("rajnishe/reside-6k")
            print(f"Downloaded to: {path}")
            # Symlink
            os.system(f'ln -sf "{path}" RESIDE-6K')
            train_h, train_c, test_h, test_c = find_dataset()
        except Exception as e:
            print(f"Download failed: {e}")
            print("Manually download: kaggle datasets download -d rajnishe/reside-6k")
            return None, None, None, None

    if train_h:
        print(f"  Train: {train_h}")
        print(f"  Test:  {test_h or 'Not found (will use train subset)'}")

    return train_h, train_c, test_h, test_c


def compute_psnr(pred, target):
    """PSNR calculate karo (dB mein)"""
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()


def compute_ssim_approx(pred, target):
    """Simplified SSIM (proper implementation ke liye full metrics.py use karo)"""
    C1, C2 = 0.01**2, 0.03**2
    mu_pred = pred.mean(dim=[2, 3], keepdim=True)
    mu_target = target.mean(dim=[2, 3], keepdim=True)
    sigma_pred = ((pred - mu_pred)**2).mean(dim=[2, 3], keepdim=True)
    sigma_target = ((target - mu_target)**2).mean(dim=[2, 3], keepdim=True)
    sigma_both = ((pred - mu_pred) * (target - mu_target)).mean(dim=[2, 3], keepdim=True)

    ssim = ((2*mu_pred*mu_target + C1) * (2*sigma_both + C2)) / \
           ((mu_pred**2 + mu_target**2 + C1) * (sigma_pred + sigma_target + C2))
    return ssim.mean().item()


def train_and_generate(epochs=20, batch_size=8, image_size=128):
    """Real data pe train karo aur outputs generate karo"""
    print("=" * 60)
    print("IMAGE DEHAZING - Training with REAL Data")
    print("=" * 60)

    # Dataset
    train_h, train_c, test_h, test_c = download_dataset()
    if train_h is None:
        print("DATASET NAHI MILA! Exiting.")
        return

    # Load datasets
    print("\nLoading datasets...")
    train_ds = RealDehazingDataset(train_h, train_c, image_size=image_size, max_images=300)
    if test_h:
        test_ds = RealDehazingDataset(test_h, test_c, image_size=image_size, max_images=50)
    else:
        # Use last 50 from train as test
        test_ds = RealDehazingDataset(train_h, train_c, image_size=image_size, max_images=50)

    # Split train into train+val
    train_size = int(0.85 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    print(f"  Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_ds)}")

    # Model
    model = UNet(in_channels=3, out_channels=3, features=[64, 128, 256],
                 use_batch_norm=True, dropout_rate=0.3).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: U-Net ({params:,} parameters)")
    print(f"  Device: {device}")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training
    train_losses, val_losses, psnr_values, ssim_values = [], [], [], []

    print(f"\nTraining for {epochs} epochs...")
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        epoch_loss = 0
        for batch_h, batch_c in train_loader:
            batch_h, batch_c = batch_h.to(device), batch_c.to(device)
            optimizer.zero_grad()
            output = model(batch_h)
            loss = criterion(output, batch_c)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        # Validate
        model.eval()
        val_loss, val_psnr, val_ssim = 0, 0, 0
        with torch.no_grad():
            for batch_h, batch_c in val_loader:
                batch_h, batch_c = batch_h.to(device), batch_c.to(device)
                output = model(batch_h)
                val_loss += criterion(output, batch_c).item()
                val_psnr += compute_psnr(output, batch_c)
                val_ssim += compute_ssim_approx(output, batch_c)

        avg_val = val_loss / len(val_loader)
        avg_psnr = val_psnr / len(val_loader)
        avg_ssim = val_ssim / len(val_loader)
        val_losses.append(avg_val)
        psnr_values.append(avg_psnr)
        ssim_values.append(avg_ssim)
        scheduler.step(avg_val)

        elapsed = time.time() - start
        if epoch % 2 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:2d}/{epochs}] Loss: {avg_train:.4f} | Val: {avg_val:.4f} | "
                  f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | Time: {elapsed:.0f}s")

    total_time = time.time() - start
    print(f"\nTraining complete! Total: {total_time:.1f}s")

    # Get test images for visualization
    test_hazy, test_clean = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        test_dehazed = model(test_hazy.to(device)).cpu()

    # Also compute hazy→clean PSNR (baseline without model)
    baseline_psnr = compute_psnr(test_hazy, test_clean)
    baseline_ssim = compute_ssim_approx(test_hazy, test_clean)

    # Generate outputs
    print("\nGenerating output images...")
    generate_outputs(model, train_losses, val_losses, psnr_values, ssim_values,
                     test_hazy, test_dehazed, test_clean,
                     baseline_psnr, baseline_ssim, params, epochs, total_time)


def generate_outputs(model, train_losses, val_losses, psnr_values, ssim_values,
                     test_hazy, test_dehazed, test_clean,
                     baseline_psnr, baseline_ssim, params, epochs, total_time):
    """5 professional output images generate karo"""

    os.makedirs('practice/results', exist_ok=True)

    # ===== 1. LOSS CURVES =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(train_losses) + 1)

    axes[0].plot(ep, train_losses, 'b-', lw=2, label='Training Loss', marker='o', ms=4)
    axes[0].plot(ep, val_losses, 'r-', lw=2, label='Validation Loss', marker='s', ms=4)
    axes[0].fill_between(ep, train_losses, val_losses, alpha=0.1, color='purple')
    axes[0].set_xlabel('Epoch', fontsize=12); axes[0].set_ylabel('L1 Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, psnr_values, 'g-', lw=2, label='PSNR (dB)', marker='D', ms=4)
    ax2 = axes[1].twinx()
    ax2.plot(ep, ssim_values, 'm--', lw=2, label='SSIM', marker='^', ms=4)
    axes[1].set_xlabel('Epoch', fontsize=12); axes[1].set_ylabel('PSNR (dB)', fontsize=12, color='g')
    ax2.set_ylabel('SSIM', fontsize=12, color='m')
    axes[1].set_title('PSNR & SSIM Over Training', fontsize=14, fontweight='bold')
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1+lines2, labels1+labels2, fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Image Dehazing - U-Net Training Metrics (RESIDE-6K)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('practice/results/output_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [1/5] Loss curves saved")

    # ===== 2. METRICS COMPARISON =====
    final_psnr = psnr_values[-1]
    final_ssim = ssim_values[-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    bars1 = axes[0].bar(['Hazy Input\n(Baseline)', 'U-Net Output\n(Dehazed)'],
                        [baseline_psnr, final_psnr],
                        color=['#ff6b6b', '#51cf66'], edgecolor='black', lw=1.5)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('PSNR Comparison', fontsize=13, fontweight='bold')
    for bar, val in zip(bars1, [baseline_psnr, final_psnr]):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{val:.1f} dB', ha='center', fontsize=12, fontweight='bold')

    bars2 = axes[1].bar(['Hazy Input\n(Baseline)', 'U-Net Output\n(Dehazed)'],
                        [baseline_ssim, final_ssim],
                        color=['#ff6b6b', '#51cf66'], edgecolor='black', lw=1.5)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('SSIM Comparison', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 1.0)
    for bar, val in zip(bars2, [baseline_ssim, final_ssim]):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')

    improvement = final_psnr - baseline_psnr
    plt.suptitle(f'Quality Metrics - PSNR improved by {improvement:.1f} dB',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('practice/results/output_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [2/5] Metrics comparison saved")

    # ===== 3. VISUAL COMPARISON (THE MOST IMPORTANT ONE!) =====
    n = min(4, test_hazy.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))

    for col in range(n):
        # Row 0: Hazy
        hazy_np = test_hazy[col].numpy().transpose(1, 2, 0)
        axes[0, col].imshow(np.clip(hazy_np, 0, 1))
        axes[0, col].set_title(f'Hazy Input #{col+1}', fontsize=11)
        axes[0, col].axis('off')

        # Row 1: Dehazed (our model output)
        dehaz_np = test_dehazed[col].numpy().transpose(1, 2, 0)
        p = compute_psnr(test_dehazed[col:col+1], test_clean[col:col+1])
        axes[1, col].imshow(np.clip(dehaz_np, 0, 1))
        axes[1, col].set_title(f'Dehazed (PSNR: {p:.1f}dB)', fontsize=11)
        axes[1, col].axis('off')

        # Row 2: Clean (ground truth)
        clean_np = test_clean[col].numpy().transpose(1, 2, 0)
        axes[2, col].imshow(np.clip(clean_np, 0, 1))
        axes[2, col].set_title(f'Ground Truth (Clean)', fontsize=11)
        axes[2, col].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('HAZY\n(Input)', fontsize=13, fontweight='bold',
                          rotation=0, labelpad=60, va='center')
    axes[1, 0].set_ylabel('DEHAZED\n(Our Model)', fontsize=13, fontweight='bold',
                          rotation=0, labelpad=60, va='center', color='green')
    axes[2, 0].set_ylabel('CLEAN\n(Ground Truth)', fontsize=13, fontweight='bold',
                          rotation=0, labelpad=60, va='center', color='blue')

    plt.suptitle('Image Dehazing Results - U-Net on RESIDE-6K Dataset',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('practice/results/output_visual_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [3/5] Visual comparison saved")

    # ===== 4. ARCHITECTURE DIAGRAM =====
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')
    ax.set_title('U-Net Architecture for Image Dehazing', fontsize=16, fontweight='bold', pad=20)

    encoder_data = [(1.5, 5, 'RGB\n128x128', '#74b9ff'),
                    (3, 4, '64ch\n64x64', '#0984e3'),
                    (4.5, 3, '128ch\n32x32', '#6c5ce7')]
    for x, y, txt, c in encoder_data:
        ax.add_patch(plt.Rectangle((x-0.5, y-0.4), 1.2, 0.8, facecolor=c, edgecolor='black', lw=2, alpha=0.8))
        ax.text(x+0.1, y, txt, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.add_patch(plt.Rectangle((6, 1.6), 1.5, 0.8, facecolor='#d63031', edgecolor='black', lw=2, alpha=0.9))
    ax.text(6.75, 2, '512ch\n16x16\nBOTTLENECK', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    decoder_data = [(9, 3, '256ch\n32x32', '#6c5ce7'),
                    (10.5, 4, '128ch\n64x64', '#0984e3'),
                    (12, 5, '64ch\n128x128', '#74b9ff')]
    for x, y, txt, c in decoder_data:
        ax.add_patch(plt.Rectangle((x-0.5, y-0.4), 1.2, 0.8, facecolor=c, edgecolor='black', lw=2, alpha=0.8))
        ax.text(x+0.1, y, txt, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.add_patch(plt.Rectangle((12.2, 5.8), 1.2, 0.7, facecolor='#00b894', edgecolor='black', lw=2))
    ax.text(12.8, 6.15, 'RGB Output\n128x128', ha='center', va='center', fontsize=8, fontweight='bold')

    for i in range(3):
        ex, ey = encoder_data[i][0]+0.6, encoder_data[i][1]+0.4
        dx, dy = decoder_data[2-i][0]-0.5, decoder_data[2-i][1]+0.4
        ax.annotate('', xy=(dx, dy), xytext=(ex, ey),
                    arrowprops=dict(arrowstyle='->', color='#e17055', lw=2, ls='--'))
        ax.text((ex+dx)/2, (ey+dy)/2+0.3, 'skip', fontsize=7, color='#e17055', ha='center', style='italic')

    ax.text(3, 6.5, 'ENCODER\n(Compress)', ha='center', fontsize=11, fontweight='bold', color='#0984e3')
    ax.text(6.75, 0.8, 'BOTTLENECK', ha='center', fontsize=10, fontweight='bold', color='#d63031')
    ax.text(10.5, 6.5, 'DECODER\n(Reconstruct)', ha='center', fontsize=11, fontweight='bold', color='#0984e3')

    plt.tight_layout()
    plt.savefig('practice/results/output_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [4/5] Architecture diagram saved")

    # ===== 5. SUMMARY TABLE =====
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')
    ax.set_title('Image Dehazing Project - Summary', fontsize=16, fontweight='bold', pad=30)

    table_data = [
        ['Model', 'U-Net (Encoder-Decoder + Skip Connections)'],
        ['Parameters', f'{params:,}'],
        ['Loss Function', 'L1 Loss (Mean Absolute Error)'],
        ['Optimizer', 'Adam (lr=0.001) + ReduceLROnPlateau'],
        ['Dataset', 'RESIDE-6K (Real Hazy-Clean Pairs)'],
        ['Image Size', '128 x 128 pixels'],
        ['Batch Size', '8'],
        ['Epochs', f'{epochs}'],
        ['Training Time', f'{total_time:.1f} seconds'],
        ['Final Train Loss', f'{train_losses[-1]:.4f}'],
        ['Final Val Loss', f'{val_losses[-1]:.4f}'],
        ['Baseline PSNR', f'{baseline_psnr:.2f} dB (hazy input)'],
        ['Final PSNR', f'{psnr_values[-1]:.2f} dB (+{psnr_values[-1]-baseline_psnr:.1f} dB improvement)'],
        ['Baseline SSIM', f'{baseline_ssim:.3f} (hazy input)'],
        ['Final SSIM', f'{ssim_values[-1]:.3f}'],
        ['Device', str(device)],
        ['Framework', 'PyTorch'],
    ]

    table = ax.table(cellText=table_data, colLabels=['Parameter', 'Value'],
                     loc='center', cellLoc='left', colWidths=[0.3, 0.6])
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)

    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[i, j]
            if i == 0:
                cell.set_facecolor('#2d3436'); cell.set_text_props(color='white', fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#dfe6e9')
            # Highlight improvements
            if i in [13, 15] and j == 1:
                cell.set_text_props(color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('practice/results/output_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  [5/5] Summary table saved")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generating professional output (REAL data)...")
    print("=" * 60 + "\n")

    train_and_generate(epochs=20, batch_size=8, image_size=128)

    print(f"\n{'='*60}")
    print(f"ALL OUTPUTS GENERATED!")
    print(f"{'='*60}")
    print(f"\nFiles in: practice/results/")
    print(f"  1. output_loss_curves.png         - Loss + PSNR/SSIM curves")
    print(f"  2. output_metrics_comparison.png   - Before vs After (real improvement)")
    print(f"  3. output_visual_comparison.png    - Hazy vs Dehazed vs Clean (REAL images!)")
    print(f"  4. output_architecture.png         - U-Net architecture diagram")
    print(f"  5. output_summary_table.png        - Complete project summary")
