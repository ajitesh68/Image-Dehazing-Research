import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return (self.fc_mu(h), self.fc_logvar(h))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return (recon, mu, log_var)

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss
if __name__ == '__main__':
    print('=' * 60)
    print('PRACTICE 08: Variational Autoencoder (VAE)')
    print('=' * 60)
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./practice/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data = datasets.MNIST('./practice/data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'\nTotal Parameters: {total_params:,}')
    print(f'Latent Dimension: 20 (784 pixels → 20 numbers mein compress!)')
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            optimizer.zero_grad()
            recon, mu, log_var = model(images)
            loss = vae_loss(recon, images, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_data)
        if epoch % 2 == 0 or epoch == 1:
            print(f'  Epoch [{epoch}/10] Average Loss: {avg_loss:.2f}')
    print('\nNayi images generate ho rahi hain random z se...')
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, 20)
        generated = model.decode(z).view(-1, 1, 28, 28)
    os.makedirs('practice/results', exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('VAE Generated Digits (random latent vectors se)', fontsize=14)
    for i in range(16):
        axes[i // 8, i % 8].imshow(generated[i].squeeze().numpy(), cmap='gray')
        axes[i // 8, i % 8].axis('off')
    plt.savefig('practice/results/08_vae_generated.png', dpi=100)
    plt.close()
    print('Latent space interpolation...')
    images, _ = next(iter(test_loader))
    with torch.no_grad():
        mu1, _ = model.encode(images[0:1].view(1, -1))
        mu2, _ = model.encode(images[1:2].view(1, -1))
    fig, axes = plt.subplots(1, 10, figsize=(16, 2))
    fig.suptitle('Latent Space Interpolation (Image 1 se Image 2 tak smooth!)', fontsize=11)
    for i, alpha in enumerate([j / 9 for j in range(10)]):
        z = (1 - alpha) * mu1 + alpha * mu2
        with torch.no_grad():
            img = model.decode(z).view(28, 28)
        axes[i].imshow(img.numpy(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'{alpha:.1f}')
    plt.savefig('practice/results/08_vae_interpolation.png', dpi=100)
    plt.close()
    print(f'\nResults saved: practice/results/08_vae_*.png')
    print('\n' + '=' * 60)
    print('PRACTICE 08 COMPLETE!')
    print('=' * 60)
    print('\nKYA SEEKHA (What You Learned):')
    print('  1. VAE = Encoder -> (mu, log_var) -> Reparameterize -> Decoder')
    print('  2. Reparameterization Trick (z = mu + std * eps)')
    print('     → Random part (eps) FIXED, learnable parts (mu,std) mein gradient flow!')
    print('  3. KL Divergence (distribution ko N(0,1) ke close rakho)')
    print('  4. Latent Space Interpolation (images ke beech smooth transitions!)')
    print('  5. VAE vs GAN:')
    print('     VAE: stable training, blurry images, smooth latent space')
    print('     GAN: unstable training, sharp images, no latent space guarantee')
    print('\nINTERVIEW QUESTION:')
    print('  Q: VAE loss ke do components kya hain?')
    print('  A: Reconstruction Loss (image quality) + KL Divergence (latent regularization)')
    print('\nNEXT: python practice/09_depthwise_separable_conv.py')