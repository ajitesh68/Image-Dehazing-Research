import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

class ConvAutoencoder(nn.Module):

    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0, output_padding=0), nn.ReLU(), nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(), nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_and_evaluate():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./practice/data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./practice/data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    model = ConvAutoencoder()
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'🧠 Conv Autoencoder — Parameters: {total_params:,}')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = []
    print(f'\n🏋️ Training — 10 epochs')
    for epoch in range(1, 11):
        epoch_loss = 0
        for images, _ in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if epoch % 2 == 0 or epoch == 1:
            print(f'  Epoch [{epoch}/10] — Loss: {avg_loss:.6f}')
    model.eval()
    test_images, _ = next(iter(test_loader))
    with torch.no_grad():
        reconstructed = model(test_images)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle('Conv Autoencoder: Original (upar) vs Reconstructed (neeche)', fontsize=14)
    for i in range(8):
        axes[0, i].imshow(test_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].squeeze().numpy(), cmap='gray')
        axes[1, i].axis('off')
    os.makedirs('practice/results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('practice/results/02_conv_autoencoder.png', dpi=100)
    plt.close()
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Conv Autoencoder — Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('practice/results/02_loss_curve.png', dpi=100)
    plt.close()
    os.makedirs('practice/saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'practice/saved_models/02_conv_autoencoder.pth')
    print(f'\n📸 Results: practice/results/02_conv_autoencoder.png')
    print(f'📈 Loss:    practice/results/02_loss_curve.png')
    print(f'💾 Model:   practice/saved_models/02_conv_autoencoder.pth')
if __name__ == '__main__':
    print('=' * 60)
    print('🎓 PRACTICE 02: Convolutional Autoencoder')
    print('=' * 60)
    print('\n📝 NAYA CONCEPT: nn.Conv2d (2D Convolution)')
    print('   Practice 01 = Dense layers (images ke liye BURA)')
    print('   Practice 02 = Conv layers (images ke liye BEST!)')
    train_and_evaluate()
    print('\n' + '=' * 60)
    print('✅ PRACTICE 02 COMPLETE!')
    print('=' * 60)
    print('\n📝 WHAT YOU LEARNED:')
    print('  1. Conv2d — spatial patterns detect karna (3×3 filter)')
    print('  2. MaxPool2d — size choti karna (downsampling)')
    print('  3. ConvTranspose2d — size badi karna (upsampling)')
    print('  4. Dense vs Conv: 10× kam params, BETTER results!')
    print('  5. Shape tracking — har layer ke baad size kya hai')
    print('\n❓ COMPARE KARO:')
    print('  Practice 01 (Dense): ~250K params, blurry reconstruction')
    print('  Practice 02 (Conv):  ~30K params, SHARPER reconstruction!')
    print('  → LESSON: Conv2d images ke liye HAMESHA better hai!')
    print('\n👉 NEXT: python practice/03_unet_from_scratch.py')
    print('   (Conv Autoencoder + SKIP CONNECTIONS = U-Net!)')