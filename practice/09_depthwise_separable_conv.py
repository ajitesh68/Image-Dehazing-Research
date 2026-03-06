import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LightweightNet(nn.Module):

    def __init__(self):
        super(LightweightNet, self).__init__()
        self.net = nn.Sequential(DepthwiseSeparableConv(1, 16), nn.BatchNorm2d(16), nn.ReLU(), DepthwiseSeparableConv(16, 32), nn.BatchNorm2d(32), nn.ReLU(), DepthwiseSeparableConv(32, 16), nn.BatchNorm2d(16), nn.ReLU(), DepthwiseSeparableConv(16, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

class HeavyNet(nn.Module):

    def __init__(self):
        super(HeavyNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)
if __name__ == '__main__':
    print('=' * 60)
    print('PRACTICE 09: Depthwise Separable Convolution')
    print('Efficient Models — MobileNet ka Secret!')
    print('=' * 60)
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./practice/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data = datasets.MNIST('./practice/data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    heavy = HeavyNet()
    light = LightweightNet()
    heavy_params = sum((p.numel() for p in heavy.parameters()))
    light_params = sum((p.numel() for p in light.parameters()))
    print(f"\n{'=' * 40}")
    print(f'PARAMETER COMPARISON:')
    print(f"{'=' * 40}")
    print(f'  Normal Conv2d:       {heavy_params:,} parameters')
    print(f'  Depthwise Separable: {light_params:,} parameters')
    print(f'  Reduction:           {heavy_params / light_params:.1f}x KAM parameters!')
    print(f'  Ye {heavy_params - light_params:,} parameters ki bachat hai!')
    print(f"{'=' * 40}")
    print('\nTraining dono models (3 epochs each)...')
    for name, model in [('Heavy (Normal Conv)', heavy), ('Light (Depthwise Sep)', light)]:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        start_time = time.time()
        for epoch in range(1, 4):
            for images, _ in train_loader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, images)
                loss.backward()
                optimizer.step()
        elapsed = time.time() - start_time
        print(f'  {name}: 3 epochs mein {elapsed:.1f}s, final loss: {loss.item():.6f}')
    print(f"\n{'=' * 40}")
    print('SPEED COMPARISON (Inference):')
    print(f"{'=' * 40}")
    x = torch.randn(100, 1, 28, 28)
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            heavy(x)
    heavy_time = time.time() - start
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            light(x)
    light_time = time.time() - start
    print(f'  Normal Conv:    {heavy_time:.3f}s (50 batches x 100 images)')
    print(f'  Depthwise Sep:  {light_time:.3f}s')
    print(f'  Speedup:        {heavy_time / light_time:.1f}x FASTER!')
    print(f"{'=' * 40}")
    os.makedirs('practice/results', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars1 = axes[0].bar(['Normal Conv', 'Depthwise Sep'], [heavy_params, light_params], color=['#FF6B6B', '#51CF66'])
    axes[0].set_title('Parameters Count', fontsize=14)
    axes[0].set_ylabel('Total Parameters')
    for bar, val in zip(bars1, [heavy_params, light_params]):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{val:,}', ha='center', va='bottom', fontweight='bold')
    bars2 = axes[1].bar(['Normal Conv', 'Depthwise Sep'], [heavy_time, light_time], color=['#FF6B6B', '#51CF66'])
    axes[1].set_title('Inference Time (seconds)', fontsize=14)
    axes[1].set_ylabel('Seconds')
    for bar, val in zip(bars2, [heavy_time, light_time]):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{val:.3f}s', ha='center', va='bottom', fontweight='bold')
    plt.suptitle('Normal Conv2d vs Depthwise Separable Conv\n(Same Architecture, HUGE Difference!)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('practice/results/09_depthwise_comparison.png', dpi=100)
    plt.close()
    print(f'\nResults: practice/results/09_depthwise_comparison.png')
    print('\n' + '=' * 60)
    print('PRACTICE 09 COMPLETE!')
    print('=' * 60)
    print('\nKYA SEEKHA (What You learned):')
    print('  1. Depthwise Conv: groups=in_channels → har channel ALAG se filter')
    print('  2. Pointwise Conv: 1x1 kernel → channels MIX karo + count badlo')
    print('  3. Depthwise Sep = Depthwise + Pointwise → 7x KAM parameters!')
    print('  4. MobileNet ka core idea — phones pe DL chalaana')
    print('  5. Speed + Memory dono mein BAHUT improvement, quality lagbhag SAME!')
    print('\nINTERVIEW QUESTIONS:')
    print('  Q1: Depthwise separable conv normal conv se kaise different hai?')
    print('  A1: Normal mein sab channels ek saath process hote hain.')
    print('      Depthwise mein pehle har channel alag se (depthwise),')
    print('      phir 1x1 se channels mix (pointwise). ~7x kam params!')
    print('  Q2: groups parameter kya karta hai PyTorch Conv2d mein?')
    print('  A2: groups channels ko separate groups mein baant deta hai.')
    print('      groups=1: normal conv. groups=in_channels: depthwise conv.')
    print('\nNEXT: python practice/10_training_tricks.py')