import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class Generator(nn.Module):

    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.Linear(latent_dim, 256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256), nn.Linear(256, 512), nn.LeakyReLU(0.2), nn.BatchNorm1d(512), nn.Linear(512, 784), nn.Tanh())

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

def train_gan(epochs=20, latent_dim=100, lr=0.0002):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = datasets.MNIST('./practice/data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    G = Generator(latent_dim)
    D = Discriminator()
    criterion = nn.BCELoss()
    g_optim = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optim = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    g_losses, d_losses = ([], [])
    print(f'\n🏋️ GAN Training — {epochs} epochs')
    for epoch in range(1, epochs + 1):
        g_loss_total, d_loss_total = (0, 0)
        for real_images, _ in loader:
            batch_size = real_images.size(0)
            real_images = real_images.view(batch_size, -1)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            z = torch.randn(batch_size, latent_dim)
            fake_images = G(z).detach()
            d_real = D(real_images)
            d_fake = D(fake_images)
            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            z = torch.randn(batch_size, latent_dim)
            fake_images = G(z)
            d_fake = D(fake_images)
            g_loss = criterion(d_fake, real_labels)
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()
        g_avg = g_loss_total / len(loader)
        d_avg = d_loss_total / len(loader)
        g_losses.append(g_avg)
        d_losses.append(d_avg)
        if epoch % 5 == 0 or epoch == 1:
            print(f'  Epoch [{epoch}/{epochs}] G_Loss: {g_avg:.4f}, D_Loss: {d_avg:.4f}')
    G.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim)
        generated = G(z).view(-1, 1, 28, 28)
    os.makedirs('practice/results', exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('GAN Generated Digits (FAKE images! Generator ne banaye!)', fontsize=14)
    for i in range(16):
        r, c = (i // 8, i % 8)
        img = generated[i].squeeze().numpy() * 0.5 + 0.5
        axes[r, c].imshow(img, cmap='gray')
        axes[r, c].axis('off')
    plt.savefig('practice/results/05_gan_generated.png', dpi=100)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, 'b-', label='Generator', linewidth=2)
    plt.plot(d_losses, 'r-', label='Discriminator', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training — Generator vs Discriminator')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('practice/results/05_gan_losses.png', dpi=100)
    plt.close()
    print(f'\n📸 Generated digits: practice/results/05_gan_generated.png')
    print(f'📈 Loss curves:      practice/results/05_gan_losses.png')
if __name__ == '__main__':
    print('=' * 60)
    print('🎓 PRACTICE 05: Basic GAN (Generative Adversarial Network)')
    print('=' * 60)
    train_gan(epochs=20)
    print('\n✅ PRACTICE 05 COMPLETE!')
    print('\n📝 WHAT YOU LEARNED:')
    print('  1. Generator — noise se images CREATE karna')
    print('  2. Discriminator — real vs fake JUDGE karna')
    print('  3. Adversarial training — alternating updates')
    print('  4. BCE Loss — binary classification ke liye')
    print('  5. .detach() — gradient flow control')
    print('  6. LeakyReLU, Tanh, BatchNorm1d')
    print('\n👉 NEXT: python practice/06_transfer_learning.py')
    print('   (Pre-trained models use karna — industry ka STANDARD!)')