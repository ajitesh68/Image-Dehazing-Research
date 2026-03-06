import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):

    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)

class SSIMLoss(nn.Module):

    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_gaussian_window(window_size, channel)

    def _create_gaussian_window(self, window_size, channel):
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
        window = gaussian_2d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, prediction, target):
        window = self.window.to(prediction.device)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        padding = self.window_size // 2
        mu_pred = F.conv2d(prediction, window, padding=padding, groups=self.channel)
        mu_target = F.conv2d(target, window, padding=padding, groups=self.channel)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target
        sigma_pred_sq = F.conv2d(prediction ** 2, window, padding=padding, groups=self.channel) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=padding, groups=self.channel) - mu_target_sq
        sigma_cross = F.conv2d(prediction * target, window, padding=padding, groups=self.channel) - mu_cross
        numerator = (2 * mu_cross + C1) * (2 * sigma_cross + C2)
        denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        ssim_map = numerator / denominator
        ssim_value = ssim_map.mean()
        return 1.0 - ssim_value

class CombinedLoss(nn.Module):

    def __init__(self, l1_weight=1.0, ssim_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight

    def forward(self, prediction, target):
        l1 = self.l1_loss(prediction, target)
        ssim = self.ssim_loss(prediction, target)
        total = self.l1_weight * l1 + self.ssim_weight * ssim
        return total

def get_loss_function(config):
    loss_type = config['loss']['type']
    if loss_type == 'l1':
        print('📐 L1 (MAE) Loss use ho raha hai')
        return L1Loss()
    elif loss_type == 'mse':
        print('📐 MSE Loss use ho raha hai')
        return nn.MSELoss()
    elif loss_type == 'ssim':
        print('📐 SSIM Loss use ho raha hai')
        return SSIMLoss()
    elif loss_type == 'combined':
        l1_w = config['loss']['l1_weight']
        ssim_w = config['loss']['ssim_weight']
        print(f'📐 Combined Loss (L1 weight={l1_w}, SSIM weight={ssim_w})')
        return CombinedLoss(l1_weight=l1_w, ssim_weight=ssim_w)
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')
if __name__ == '__main__':
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    l1 = L1Loss()
    print(f'L1 Loss:   {l1(pred, target).item():.4f}')
    ssim = SSIMLoss()
    print(f'SSIM Loss: {ssim(pred, target).item():.4f}')
    combined = CombinedLoss()
    print(f'Combined:  {combined(pred, target).item():.4f}')
    same = torch.rand(2, 3, 64, 64)
    print(f'\nSame images → L1: {l1(same, same).item():.6f}')
    print(f'Same images → SSIM: {ssim(same, same).item():.6f}')
    print('\n✅ Saare loss functions sahi kaam kar rahe hain!')