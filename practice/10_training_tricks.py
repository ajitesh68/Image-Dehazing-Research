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
import copy

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

def demo_lr_schedulers():
    epochs = 100
    model = SimpleNet()
    schedulers = {}
    opt = optim.Adam(model.parameters(), lr=0.01)
    s = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]['lr'])
        s.step()
    schedulers['Step Decay'] = lrs
    opt = optim.Adam(model.parameters(), lr=0.01)
    s = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]['lr'])
        s.step()
    schedulers['Cosine Annealing'] = lrs
    opt = optim.Adam(model.parameters(), lr=0.001)
    s = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=epochs)
    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]['lr'])
        s.step()
    schedulers['OneCycle'] = lrs
    base_lr = 0.01
    warmup_epochs = 10
    lrs = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
        else:
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        lrs.append(lr)
    schedulers['Warmup + Cosine'] = lrs
    os.makedirs('practice/results', exist_ok=True)
    plt.figure(figsize=(12, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#A855F7']
    for (name, lrs), color in zip(schedulers.items(), colors):
        plt.plot(lrs, label=name, linewidth=2.5, color=color)
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Learning Rate', fontsize=13)
    plt.title('LR Schedulers Comparison - Kaunsa Best Hai?', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('practice/results/10_lr_schedulers.png', dpi=100, bbox_inches='tight')
    plt.close()
    print('LR Scheduler comparison saved! (practice/results/10_lr_schedulers.png)')

class EarlyStopping:

    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            print(f'    EarlyStopping: {self.counter}/{self.patience} (no improvement for {self.counter} epochs)')
            if self.counter >= self.patience:
                self.should_stop = True
                print(f'    STOPPING! {self.patience} epochs se improvement nahi hui.')
                print(f'    Best validation loss tha: {self.best_loss:.6f}')

def demo_weight_init():
    print(f"\n{'=' * 40}")
    print('WEIGHT INITIALIZATION COMPARISON:')
    print(f"{'=' * 40}")
    model = SimpleNet()

    def count_dead_neurons(m, x):
        with torch.no_grad():
            out = m(x)
            dead = (out == 0).float().mean().item() * 100
        return dead
    x = torch.randn(10, 1, 28, 28)
    dead_default = count_dead_neurons(model, x)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    dead_kaiming = count_dead_neurons(model, x)
    print(f'  Default init  — dead neurons: {dead_default:.1f}%')
    print(f'  Kaiming init  — dead neurons: {dead_kaiming:.1f}%')
    print(f'  (Kam dead neurons = better gradient flow = faster training!)')
    print(f"{'=' * 40}")

def demo_gradient_clipping():
    print(f"\n{'=' * 40}")
    print('GRADIENT CLIPPING DEMO:')
    print(f"{'=' * 40}")
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(4, 1, 28, 28)
    output = model(x)
    loss = ((output - x) ** 2).mean()
    loss.backward()
    grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
    optimizer.zero_grad()
    output = model(x)
    loss = ((output - x) ** 2).mean()
    loss.backward()
    grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f'  Before clipping: gradient norm = {grad_norm_before:.4f}')
    print(f'  After clipping:  gradient norm = {min(grad_norm_after, 1.0):.4f}')
    print(f'  max_norm = 1.0 (isse zyada nahi hoga)')
    print(f'  (Clipping se training STABLE hoti hai — koi sudden crash nahi!)')
    print(f"{'=' * 40}")
if __name__ == '__main__':
    print('=' * 60)
    print('PRACTICE 10: Training Tricks')
    print('(LR Schedulers, Early Stopping, Weight Init, Grad Clip)')
    print('=' * 60)
    print('\n[1/3] Learning Rate Schedulers...')
    demo_lr_schedulers()
    print('\n[2/3] Weight Initialization...')
    demo_weight_init()
    print('\n[3/3] Gradient Clipping...')
    demo_gradient_clipping()
    print('\n' + '=' * 60)
    print('PRACTICE 10 COMPLETE!')
    print('=' * 60)
    print('\nKYA SEEKHA (What You Learned):')
    print('  1. LR Schedulers:')
    print('     - Step Decay (har N epochs pe drop karo)')
    print('     - Cosine Annealing (smooth S-curve decrease)')
    print('     - OneCycle (pehle badhao phir ghataao — FASTEST training!)')
    print('     - Warmup + Cosine (Transformers ke liye MANDATORY!)')
    print('  2. Early Stopping:')
    print('     - Validation loss improve nahi ho rahi → STOP!')
    print('     - patience=7, delta=0.001 → common settings')
    print('  3. Weight Initialization:')
    print('     - Kaiming/He → ReLU ke liye BEST')
    print('     - Xavier/Glorot → Sigmoid/Tanh ke liye')
    print('     - Zeros se KABHI init mat karo!')
    print('  4. Gradient Clipping:')
    print('     - clip_grad_norm_(params, max_norm=1.0)')
    print('     - Exploding gradients se bachao!')
    print('\n' + '=' * 60)
    print('PRACTICE LAB COMPLETE! 10/10 FILES!')
    print('=' * 60)
    print('\nTumne DL ke SAARE IMPORTANT concepts samajh liye:')
    print('  01: Dense Autoencoder (nn.Linear, encoder-decoder)')
    print('  02: Conv Autoencoder (Conv2d, MaxPool, ConvTranspose)')
    print('  03: ResNet Blocks (skip connections, vanishing gradient)')
    print('  04: Attention (Channel, Spatial, CBAM)')
    print('  05: GAN (Generator, Discriminator, adversarial)')
    print('  06: Transfer Learning (pre-trained, fine-tuning)')
    print('  07: Transformer (Self-attention, ViT, Q/K/V)')
    print('  08: VAE (reparameterization, KL divergence)')
    print('  09: Depthwise Separable Conv (MobileNet, efficiency)')
    print('  10: Training Tricks (LR schedule, early stop, grad clip)')
    print('\nAB RESEARCH + DEHAZING IMPROVEMENT PE KAAM KARO!')
    print('Tumhara FOUNDATION ready hai — ab kuch BHI bana sakte ho!')