import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.shape
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_weights = self.fc(avg_out)
        max_weights = self.fc(max_out)
        weights = (avg_weights + max_weights) / 2
        weights = weights.view(b, c, 1, 1)
        return x * weights

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(combined))
        return x * attention_map

class CBAM(nn.Module):

    def __init__(self, channels, reduction=4, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionNet(nn.Module):

    def __init__(self):
        super(AttentionNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), CBAM(32), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), CBAM(32), nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

class PlainNet(nn.Module):

    def __init__(self):
        super(PlainNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

def train_model(model, loader, epochs=5, name='Model'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for epoch in range(1, epochs + 1):
        total = 0
        for images, _ in loader:
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, images)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        losses.append(avg)
        if epoch % 2 == 0 or epoch == 1:
            print(f'  {name} — Epoch [{epoch}/{epochs}] Loss: {avg:.6f}')
    return losses
if __name__ == '__main__':
    print('=' * 60)
    print('🎓 PRACTICE 04: Attention Mechanism (CBAM)')
    print('=' * 60)
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./practice/data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./practice/data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    print('\n🔴 Training Plain Network (BINA attention)...')
    plain = PlainNet()
    plain_losses = train_model(plain, train_loader, epochs=5, name='Plain')
    print('\n🟢 Training CBAM Network (Attention ke saath)...')
    attn = AttentionNet()
    attn_losses = train_model(attn, train_loader, epochs=5, name='CBAM')
    print(f'\n📊 ATTENTION ka FARK:')
    print(f'   Plain params:  {sum((p.numel() for p in plain.parameters())):,}')
    print(f'   CBAM params:   {sum((p.numel() for p in attn.parameters())):,}')
    print(f'   Plain loss:    {plain_losses[-1]:.6f}')
    print(f'   CBAM loss:     {attn_losses[-1]:.6f}')
    os.makedirs('practice/results', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(plain_losses, 'r--', label='Plain', linewidth=2)
    plt.plot(attn_losses, 'g-', label='CBAM Attention', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Attention vs No Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('practice/results/04_attention_comparison.png', dpi=100)
    plt.close()
    print(f'\n📸 Results: practice/results/04_attention_comparison.png')
    print('\n✅ PRACTICE 04 COMPLETE!')
    print('\n📝 WHAT YOU LEARNED:')
    print('  1. Channel Attention (SE Block) — kaun se features important')
    print('  2. Spatial Attention — kahan focus karna hai')
    print('  3. CBAM = Channel + Spatial combined')
    print('  4. AdaptiveAvgPool2d — universal pooling')
    print('  5. Attention ko kisi bhi network mein plug karna!')
    print('\n👉 NEXT: python practice/05_basic_gan.py')
    print('   (Generator vs Discriminator — images GENERATE karna seekho!)')