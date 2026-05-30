import argparse
import yaml
import torch

from data.dataset import smart_create_dataloaders
from models.unet import UNet
from models.metrics import calculate_psnr, calculate_ssim, AverageMeter
from utils.visualization import save_comparison


def evaluate(model, test_loader, device, save_results=True, output_dir='experiments'):
    model.eval()

    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    print("\n📊 Test set pe evaluation ho rahi hai...")

    with torch.no_grad():
        for batch_idx, (hazy, clean) in enumerate(test_loader):
            hazy = hazy.to(device)
            clean = clean.to(device)

            prediction = model(hazy)

            for i in range(hazy.size(0)):
                psnr = calculate_psnr(prediction[i:i+1], clean[i:i+1])
                ssim = calculate_ssim(prediction[i:i+1], clean[i:i+1])
                psnr_meter.update(psnr)
                ssim_meter.update(ssim)

            if save_results and batch_idx == 0:
                save_comparison(
                    hazy[:4], prediction[:4], clean[:4],
                    save_path=f'{output_dir}/test_results.png'
                )

    results = {
        'psnr': psnr_meter.avg,
        'ssim': ssim_meter.avg
    }

    print(f"\n{'='*40}")
    print(f"  TEST RESULTS")
    print(f"  PSNR: {results['psnr']:.2f} dB")
    print(f"  SSIM: {results['ssim']:.4f}")
    print(f"{'='*40}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Dehazing Model Evaluate')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, default='experiments/best_model.pth',
                        help='Saved model checkpoint ka path')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = smart_create_dataloaders(config)

    model = UNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        features=config['model']['features'],
        use_batch_norm=config['model']['use_batch_norm'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"(Val loss: {checkpoint['val_loss']:.4f}, "
          f"Val PSNR: {checkpoint['val_psnr']:.2f})")

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()