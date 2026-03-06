import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.block(x)
        output = residual + x
        output = self.relu(output)
        return output

class ResNetReconstructor(nn.Module):

    def __init__(self, num_res_blocks=4):
        super(ResNetReconstructor, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(32))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.decoder = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

class PlainNetwork(nn.Module):

    def __init__(self, num_blocks=4):
        super(PlainNetwork, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        plain_blocks = []
        for _ in range(num_blocks):
            plain_blocks.extend([nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True)])
        self.blocks = nn.Sequential(*plain_blocks)
        self.decoder = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        x = self.decoder(x)
        return x

def train_model(model, train_loader, epochs=5, name='Model'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        for images, _ in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(train_loader)
        losses.append(avg)
        print(f'  {name} — Epoch [{epoch}/{epochs}] Loss: {avg:.6f}')
    return losses
if __name__ == '__main__':
    print('=' * 60)
    print('🎓 PRACTICE 03: ResNet Blocks (Residual Connections)')
    print('=' * 60)
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./practice/data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./practice/data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    print('\n🔴 Training Plain Network (BINA skip connections)...')
    plain = PlainNetwork(num_blocks=4)
    plain_losses = train_model(plain, train_loader, epochs=5, name='Plain')
    print('\n🟢 Training ResNet (SKIP connections ke saath)...')
    resnet = ResNetReconstructor(num_res_blocks=4)
    resnet_losses = train_model(resnet, train_loader, epochs=5, name='ResNet')
    print(f'\n📊 COMPARISON:')
    print(f'   Plain final loss:  {plain_losses[-1]:.6f}')
    print(f'   ResNet final loss: {resnet_losses[-1]:.6f}')
    if resnet_losses[-1] < plain_losses[-1]:
        diff = (plain_losses[-1] - resnet_losses[-1]) / plain_losses[-1] * 100
        print(f'   → ResNet {diff:.1f}% BETTER! Skip connections ka kamaal! 🏆')
    os.makedirs('practice/results', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(plain_losses, 'r--', label='Plain (no skip)', linewidth=2)
    plt.plot(resnet_losses, 'g-', label='ResNet (skip connections)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Skip Connections Ka Fark!')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('practice/results/03_resnet_comparison.png', dpi=100)
    plt.close()
    resnet.eval()
    images, _ = next(iter(test_loader))
    with torch.no_grad():
        recon = resnet(images)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('ResNet Reconstructor: Original vs Reconstructed', fontsize=14)
    for i in range(8):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].squeeze().numpy(), cmap='gray')
        axes[1, i].axis('off')
    plt.savefig('practice/results/03_resnet_results.png', dpi=100)
    plt.close()
    print(f'\n📸 Results: practice/results/03_resnet_*.png')
    print('\n✅ PRACTICE 03 COMPLETE!')
    print('\n📝 WHAT YOU LEARNED:')
    print('  1. Residual Block: output = F(x) + x')
    print('  2. Skip connections gradient flow GUARANTEE karte hain')
    print('  3. Plain vs ResNet comparison — skip connections WIN!')
    print('  4. ResNet 2015 paper — 200K+ citations!')
    print('\n👉 NEXT: python practice/04_attention_mechanism.py')