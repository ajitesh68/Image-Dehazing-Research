import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math

class SelfAttention(nn.Module):

    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return (output, attention_weights)

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, ff_dim=128, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        normed = self.norm1(x)
        attended, weights = self.attention(normed)
        x = x + attended
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        return (x, weights)

class SimpleViT(nn.Module):

    def __init__(self, image_size=28, patch_size=7, embed_dim=64, num_classes=10, num_blocks=2):
        super(SimpleViT, self).__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(num_blocks)])
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, -1, self.patch_size * self.patch_size)
        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens, _ = block(tokens)
        cls_output = tokens[:, 0]
        logits = self.classifier(cls_output)
        return logits
if __name__ == '__main__':
    print('=' * 60)
    print('🎓 PRACTICE 07: Transformer (Self-Attention + ViT)')
    print('=' * 60)
    transform = transforms.ToTensor()
    train_data = datasets.MNIST('./practice/data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./practice/data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    model = SimpleViT(image_size=28, patch_size=7, embed_dim=64, num_classes=10, num_blocks=2)
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'\n🧠 ViT Parameters: {total_params:,}')
    print(f'   Patches: 28/7 = 4×4 = 16 patches + 1 CLS = 17 tokens')
    print(f'   Transformer blocks: 2')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f'\n🏋️ Training — 5 epochs')
    losses = []
    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        losses.append(avg)
        print(f'  Epoch [{epoch}/5] Loss: {avg:.4f}')
    model.eval()
    correct, total = (0, 0)
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'\n📊 Test Accuracy: {accuracy:.1f}%')
    os.makedirs('practice/results', exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Vision Transformer — MNIST (Accuracy: {accuracy:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('practice/results/07_transformer_loss.png', dpi=100)
    plt.close()
    print(f'\n📈 Loss curve: practice/results/07_transformer_loss.png')
    print('\n✅ PRACTICE 07 COMPLETE!')
    print('\n📝 WHAT YOU LEARNED:')
    print('  1. Self-Attention — Q, K, V mechanism')
    print('  2. Scaled dot-product attention (÷ √d_k)')
    print('  3. TransformerBlock — attention + FFN + residual + LayerNorm')
    print('  4. Vision Transformer (ViT) — patches + CLS token + position embed')
    print('  5. GELU activation')
    print('  6. LayerNorm vs BatchNorm')
    print('\n🎉 PRACTICE LAB COMPLETE!')
    print('   Ab tum SAARI major DL architectures samajhte ho!')
    print('   → Autoencoder, Conv, ResNet, Attention, GAN, Transfer, Transformer')
    print('   → Ab RESEARCH aur DEHAZING improvements pe kaam karo! 🚀')