from google.colab import drive
drive.mount('/content/drive')
import shutil
import os
PROJECT_PATH = '/content/drive/MyDrive/image-dehazing-research'
DATASET_PATH = '/content/drive/MyDrive/RESIDE-6K'
assert os.path.exists(PROJECT_PATH), f'Project nahi mila: {PROJECT_PATH}'
assert os.path.exists(DATASET_PATH), f'Dataset nahi mila: {DATASET_PATH}'
print('Project aur Dataset DONO mil gaye!')
import sys
os.chdir(PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)
print(f'Working directory: {os.getcwd()}')
import yaml
config = yaml.safe_load(open('configs/default.yaml'))
config['data']['data_dir'] = DATASET_PATH
config['data']['num_workers'] = 2
config['training']['batch_size'] = 16
config['data']['image_size'] = 256
config['training']['epochs'] = 50
with open('configs/colab_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Colab config saved!')
print(f'  Dataset: {DATASET_PATH}')
print(f"  Batch size: {config['training']['batch_size']}")
print(f"  Image size: {config['data']['image_size']}")
print(f"  Epochs: {config['training']['epochs']}")
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1000000000.0:.1f} GB')
else:
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print('TPU detected!')
    except:
        device = torch.device('cpu')
        print('WARNING: No GPU/TPU! Training BAHUT slow hogi.')
from data.dataset import smart_create_dataloaders
config_colab = yaml.safe_load(open('configs/colab_config.yaml'))
train_loader, val_loader, test_loader = smart_create_dataloaders(config_colab)
batch = next(iter(train_loader))
print(f'Batch shape: hazy={batch[0].shape}, clean={batch[1].shape}')
print(f'Pixel range: [{batch[0].min():.3f}, {batch[0].max():.3f}]')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(4):
    img = batch[0][i].permute(1, 2, 0).numpy()
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Hazy {i + 1}')
    axes[0, i].axis('off')
    img = batch[1][i].permute(1, 2, 0).numpy()
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'Clean {i + 1}')
    axes[1, i].axis('off')
plt.suptitle('RESIDE-6K: Hazy (upar) vs Clean (neeche)', fontsize=16)
plt.tight_layout()
plt.show()
from models.unet import UNet
from models.losses import get_loss_function
from models.metrics import calculate_psnr, calculate_ssim, AverageMeter
model = UNet(in_channels=config_colab['model']['in_channels'], out_channels=config_colab['model']['out_channels'], features=config_colab['model']['features'], use_batch_norm=config_colab['model']['use_batch_norm'], dropout_rate=config_colab['model']['dropout_rate']).to(device)
total_params = sum((p.numel() for p in model.parameters()))
print(f'Model parameters: {total_params:,}')
criterion = get_loss_function(config_colab)
optimizer = torch.optim.Adam(model.parameters(), lr=config_colab['training']['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
epochs = config_colab['training']['epochs']
best_psnr = 0
train_losses, val_psnrs = ([], [])
print(f'\nTraining shuru! {epochs} epochs, device={device}')
print('=' * 50)
for epoch in range(1, epochs + 1):
    model.train()
    loss_meter = AverageMeter()
    for batch_idx, (hazy, clean) in enumerate(train_loader):
        hazy, clean = (hazy.to(device), clean.to(device))
        optimizer.zero_grad()
        output = model(hazy)
        loss = criterion(output, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_meter.update(loss.item())
    train_losses.append(loss_meter.avg)
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    with torch.no_grad():
        for hazy, clean in val_loader:
            hazy, clean = (hazy.to(device), clean.to(device))
            output = model(hazy)
            psnr = calculate_psnr(output, clean)
            ssim = calculate_ssim(output, clean)
            psnr_meter.update(psnr)
            ssim_meter.update(ssim)
    val_psnrs.append(psnr_meter.avg)
    scheduler.step(loss_meter.avg)
    if psnr_meter.avg > best_psnr:
        best_psnr = psnr_meter.avg
        os.makedirs('experiments', exist_ok=True)
        torch.save(model.state_dict(), 'experiments/best_model.pth')
    print(f'Epoch [{epoch}/{epochs}] Loss: {loss_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f} dB | SSIM: {ssim_meter.avg:.4f} | Best PSNR: {best_psnr:.2f} dB')
print(f'\nTraining complete! Best PSNR: {best_psnr:.2f} dB')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)
ax2.plot(val_psnrs, 'g-', linewidth=2)
ax2.axhline(y=best_psnr, color='r', linestyle='--', label=f'Best: {best_psnr:.2f}')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('PSNR (dB)')
ax2.set_title('Validation PSNR')
ax2.grid(True, alpha=0.3)
ax2.legend()
plt.suptitle('Image Dehazing — Training Progress', fontsize=14)
plt.tight_layout()
plt.show()
model.load_state_dict(torch.load('experiments/best_model.pth'))
model.eval()
test_batch = next(iter(test_loader))
hazy_test, clean_test = (test_batch[0].to(device), test_batch[1].to(device))
with torch.no_grad():
    dehazed = model(hazy_test)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
for i in range(4):
    axes[0, i].imshow(hazy_test[i].cpu().permute(1, 2, 0).numpy())
    axes[0, i].set_title('Hazy')
    axes[0, i].axis('off')
    axes[1, i].imshow(dehazed[i].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1, i].set_title('Dehazed')
    axes[1, i].axis('off')
    axes[2, i].imshow(clean_test[i].cpu().permute(1, 2, 0).numpy())
    axes[2, i].set_title('Ground Truth')
    axes[2, i].axis('off')
plt.suptitle(f'Dehazing Results (Best PSNR: {best_psnr:.2f} dB)', fontsize=16)
plt.tight_layout()
plt.show()
save_path = f'/content/drive/MyDrive/image-dehazing-research/experiments/best_model_psnr_{best_psnr:.1f}.pth'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f'Model saved to Drive: {save_path}')
print('Ab ye model apne laptop pe bhi use kar sakte ho!')