import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

def train_transfer_learning():
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.CIFAR10('./practice/data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./practice/data', train=False, download=True, transform=transform)
    train_subset = torch.utils.data.Subset(train_data, range(2000))
    test_subset = torch.utils.data.Subset(test_data, range(500))
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    print(f'📁 CIFAR-10: Train={len(train_subset)}, Test={len(test_subset)}')
    print('\n🔴 METHOD 1: Scratch se train (pre-trained NAHI)...')
    model_scratch = models.resnet18(weights=None)
    model_scratch.fc = nn.Linear(512, 10)
    scratch_loss, scratch_acc, scratch_time = train_and_eval(model_scratch, train_loader, test_loader, epochs=3, name='Scratch')
    print('\n🟢 METHOD 2: Transfer Learning (pre-trained weights)...')
    model_transfer = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model_transfer.parameters():
        param.requires_grad = False
    model_transfer.fc = nn.Linear(512, 10)
    transfer_loss, transfer_acc, transfer_time = train_and_eval(model_transfer, train_loader, test_loader, epochs=3, name='Transfer')
    print(f"\n{'=' * 50}")
    print(f'📊 FINAL COMPARISON:')
    print(f"{'=' * 50}")
    print(f"  {'Method':<20} {'Accuracy':>10} {'Time':>10}")
    print(f"  {'Scratch':<20} {scratch_acc:>9.1f}% {scratch_time:>9.1f}s")
    print(f"  {'Transfer Learning':<20} {transfer_acc:>9.1f}% {transfer_time:>9.1f}s")
    print(f"{'=' * 50}")
    if transfer_acc > scratch_acc:
        print(f'  → Transfer Learning {transfer_acc - scratch_acc:.1f}% BETTER! 🏆')
        print(f'  → Aur FASTER bhi! Pre-trained knowledge ka kamaal!')
    os.makedirs('practice/results', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(['Scratch', 'Transfer'], [scratch_acc, transfer_acc], color=['red', 'green'])
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax2.bar(['Scratch', 'Transfer'], [scratch_time, transfer_time], color=['red', 'green'])
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Time')
    plt.suptitle('Transfer Learning vs Training from Scratch', fontsize=14)
    plt.tight_layout()
    plt.savefig('practice/results/06_transfer_comparison.png', dpi=100)
    plt.close()
    print(f'\n📸 Results: practice/results/06_transfer_comparison.png')

def train_and_eval(model, train_loader, test_loader, epochs=3, name='Model'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'  {name} — Epoch [{epoch}/{epochs}] Loss: {total_loss / len(train_loader):.4f}')
    elapsed = time.time() - start
    model.eval()
    correct, total = (0, 0)
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return (total_loss / len(train_loader), accuracy, elapsed)
if __name__ == '__main__':
    print('=' * 60)
    print('🎓 PRACTICE 06: Transfer Learning (Pre-trained Models)')
    print('=' * 60)
    train_transfer_learning()
    print('\n✅ PRACTICE 06 COMPLETE!')
    print('\n📝 WHAT YOU LEARNED:')
    print('  1. Pre-trained models load karna (ResNet18)')
    print('  2. Layer freezing (requires_grad = False)')
    print('  3. Last layer replace karna (customize)')
    print('  4. CrossEntropyLoss — multi-class classification')
    print('  5. Transfer Learning BAHUT faster + better!')
    print('  6. ImageNet normalization values ka meaning')
    print('\n👉 NEXT: python practice/07_transformer_basics.py')
    print('   (Self-Attention — GPT/BERT ka core concept!)')