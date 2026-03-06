import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def load_mnist():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./practice/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST(root='./practice/data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f'📁 MNIST loaded!')
    print(f'   Training images: {len(train_dataset)}')
    print(f'   Test images:     {len(test_dataset)}')
    print(f'   Image size:      28×28 = 784 pixels')
    return (train_loader, test_loader)

class BasicAutoencoder(nn.Module):

    def __init__(self, input_size=784, latent_size=32):
        super(BasicAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, latent_size), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_size, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, input_size), nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train(model, train_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    print(f'\n🏋️ Training shuru! {epochs} epochs, LR={lr}')
    print('=' * 50)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.view(images.size(0), -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        if epoch % 2 == 0 or epoch == 1:
            print(f'  Epoch [{epoch}/{epochs}] — Loss: {avg_loss:.6f}')
    return losses

def visualize_results(model, test_loader, save_path='practice/results/01_autoencoder.png'):
    model.eval()
    images, _ = next(iter(test_loader))
    images_flat = images.view(images.size(0), -1)
    with torch.no_grad():
        reconstructed = model(images_flat)
    reconstructed = reconstructed.view(-1, 1, 28, 28)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('Basic Autoencoder: Original (upar) vs Reconstructed (neeche)', fontsize=14)
    for i in range(8):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        axes[1, i].imshow(reconstructed[i].squeeze().numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f'\n📸 Results saved: {save_path}')
if __name__ == '__main__':
    print('=' * 60)
    print('🎓 PRACTICE 01: Basic Autoencoder (Dense Layers)')
    print('=' * 60)
    train_loader, test_loader = load_mnist()
    model = BasicAutoencoder(input_size=784, latent_size=32)
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'\n🧠 Model parameters: {total_params:,}')
    print(f'   Encoder: 784 → 256 → 128 → 32 (compress)')
    print(f'   Decoder: 32 → 128 → 256 → 784 (reconstruct)')
    losses = train(model, train_loader, epochs=10, lr=0.001)
    visualize_results(model, test_loader)
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve — Autoencoder')
    plt.grid(True, alpha=0.3)
    plt.savefig('practice/results/01_loss_curve.png', dpi=100)
    plt.close()
    print('📈 Loss curve saved: practice/results/01_loss_curve.png')
    os.makedirs('practice/saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'practice/saved_models/01_autoencoder.pth')
    print('💾 Model saved: practice/saved_models/01_autoencoder.pth')
    print('\n' + '=' * 60)
    print('✅ PRACTICE 01 COMPLETE!')
    print('=' * 60)
    print('\n📝 WHAT YOU LEARNED:')
    print('  1. nn.Linear — fully connected layer')
    print('  2. nn.Sequential — layers chain karna')
    print('  3. Encoder-Decoder pattern (compress → reconstruct)')
    print('  4. Latent space (32 numbers mein image ka essence)')
    print('  5. MSE Loss, Adam optimizer')
    print('  6. 5-line training pattern')
    print('\n👉 NEXT: python practice/02_conv_autoencoder.py')
    print('   (nn.Linear ki jagah nn.Conv2d use karenge — images ke liye BETTER!)')