import torch
import math


def calculate_psnr(prediction, target, max_value=1.0):
    mse = torch.mean((prediction - target) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10.0 * math.log10(max_value ** 2 / mse.item())
    return psnr


def calculate_ssim(prediction, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = torch.nn.functional.avg_pool2d(prediction, kernel_size=11, stride=1, padding=5)
    mu_target = torch.nn.functional.avg_pool2d(target, kernel_size=11, stride=1, padding=5)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = torch.nn.functional.avg_pool2d(
        prediction ** 2, kernel_size=11, stride=1, padding=5
    ) - mu_pred_sq

    sigma_target_sq = torch.nn.functional.avg_pool2d(
        target ** 2, kernel_size=11, stride=1, padding=5
    ) - mu_target_sq

    sigma_cross = torch.nn.functional.avg_pool2d(
        prediction * target, kernel_size=11, stride=1, padding=5
    ) - mu_cross

    numerator = (2 * mu_cross + C1) * (2 * sigma_cross + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)

    ssim_map = numerator / denominator
    return ssim_map.mean().item()


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    print(f"Random images → PSNR: {calculate_psnr(pred, target):.2f} dB")
    print(f"Random images → SSIM: {calculate_ssim(pred, target):.4f}")

    same = torch.rand(2, 3, 64, 64)
    print(f"\nIdentical images → PSNR: {calculate_psnr(same, same):.2f} dB")
    print(f"Identical images → SSIM: {calculate_ssim(same, same):.4f}")

    meter = AverageMeter()
    meter.update(25.0)
    meter.update(27.0)
    meter.update(26.0)
    print(f"\nAverageMeter test → avg: {meter.avg:.2f} (expected: 26.00)")

    print("\n✅ Saare metrics sahi kaam kar rahe hain!")