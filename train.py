import os
import time
import yaml
import argparse

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataset import smart_create_dataloaders
from models.unet import UNet
from models.losses import get_loss_function
from models.metrics import calculate_psnr, calculate_ssim, AverageMeter


def get_device(config):
    accelerator_config = config.get('accelerator', {})
    device_choice = accelerator_config.get('device', 'auto')

    if device_choice == 'tpu':
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print("🚀 TPU detected! Using Google TPU for training.")
            return device
        except ImportError:
            print("⚠️ torch_xla not installed. Falling back to CPU/GPU...")

    if device_choice == 'cuda' or (device_choice == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda')
        print(f"🎮 GPU detected! Using: {torch.cuda.get_device_name(0)}")
        return device

    device = torch.device('cpu')
    print("💻 Using CPU (no GPU/TPU detected)")
    return device


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()

    for batch_idx, (hazy, clean) in enumerate(train_loader):
        hazy = hazy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        prediction = model(hazy)
        loss = criterion(prediction, clean)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), hazy.size(0))

        with torch.no_grad():
            psnr = calculate_psnr(prediction, clean)
            psnr_meter.update(psnr, hazy.size(0))

        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} | PSNR: {psnr:.2f} dB")

    return {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg
    }


def validate(model, val_loader, criterion, device):
    model.eval()

    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        for hazy, clean in val_loader:
            hazy = hazy.to(device)
            clean = clean.to(device)

            prediction = model(hazy)
            loss = criterion(prediction, clean)

            loss_meter.update(loss.item(), hazy.size(0))
            psnr_meter.update(calculate_psnr(prediction, clean), hazy.size(0))
            ssim_meter.update(calculate_ssim(prediction, clean), hazy.size(0))

    return {
        'loss': loss_meter.avg,
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg
    }


def main():
    parser = argparse.ArgumentParser(description='Image Dehazing Model Train Karo')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Config file ka path')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = get_device(config)

    print("\n📁 Dataset load ho raha hai...")
    train_loader, val_loader, test_loader = smart_create_dataloaders(config)

    print("\n🧠 Model ban raha hai...")
    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features'],
        use_batch_norm=config['model']['use_batch_norm'],
        dropout_rate=config['model']['dropout_rate']
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = get_loss_function(config)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training']['scheduler']['patience'],
        factor=config['training']['scheduler']['factor'],
    )

    print("\n🏋️ Training shuru ho rahi hai...")
    print("=" * 60)

    best_val_loss = float('inf')
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    epochs = config['training']['epochs']

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_metrics['loss'])

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch [{epoch}/{epochs}] ({elapsed:.1f}s)")
        print(f"  Train — Loss: {train_metrics['loss']:.4f} | PSNR: {train_metrics['psnr']:.2f} dB")
        print(f"  Val   — Loss: {val_metrics['loss']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB | SSIM: {val_metrics['ssim']:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print("-" * 60)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_path = os.path.join(save_dir, 'best_model.pth')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim'],
            }, save_path)
            print(f"  💾 Best model saved! (val_loss: {best_val_loss:.4f})")

        if epoch % config['output']['save_every'] == 0:
            save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

    print("\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()