# 🧪 Google Colab Notebook — GitHub + Kaggle Workflow
# ====================================================
# AB TUMHE KUCH BHI UPLOAD NAHI KARNA! 🎉
# Bas ye pura code copy karo, Colab mein paste karo, aur chalne do.
# ====================================================

# %% [markdown]
# # 🌫️ Image Dehazing — Training (No Uploads Required!)
# Ye notebook tumhari **GitHub repository** kheechegi aur **Kaggle** se dataset download karegi.

# %% Step 1: Tumhara Code GitHub Se Clone Karo! (1 second mein!)
import os

if not os.path.exists('/content/Image-Dehazing-Research'):
    !git clone https://github.com/ajitesh68/Image-Dehazing-Research.git
else:
    # Agar pehle se hai, toh sirf naya code pull karlo
    os.chdir('/content/Image-Dehazing-Research')
    !git pull
    os.chdir('/content')

# Project folder mein aao
os.chdir('/content/Image-Dehazing-Research')
import sys
sys.path.insert(0, '/content/Image-Dehazing-Research')
print(f"Working Directory: {os.getcwd()}")

# %% Step 2: Kaggle se dataset download karna (Bina Google Drive ke!)
# Zaroori baat: Iske liye tumhe Kaggle ka API token (kaggle.json) chahiye
#
# KAISE MILEGA?
# 1. Kaggle.com pe apne profile -> Settings mein jao
# 2. Neeche scroll karke API section mein "Create New Token" pe click karo
# 3. Ek 'kaggle.json' file download hogi. Yahan Colab panel mein usko upload kardo bas ek baar!

import shutil
import yaml

if not os.path.exists('/root/.kaggle/kaggle.json'):
    # Colab mein 'kaggle.json' upload hone ka wait karega
    from google.colab import files
    print("Kaggle account verify karne ke liye apni 'kaggle.json' file upload karo:")
    files.upload()
    
    # Kaggle directory setup karo
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    print("Kaggle API key setup ho gayi!")

# Dataset directly download aur unzip karo
DATASET_DIR = '/content/RESIDE-6K'
if not os.path.exists(DATASET_DIR):
    print("Dataset download ho raha hai Kaggle se (kuch seconds lagenge)...")
    !pip install kaggle --quiet
    # Ye command RESIDE dataset ko download karti hai (isko check kar lena ki exact dataset yahi tha)
    # Agar dataset badalna ho toh uski command kaggle se copy karke idhar paste kar dena.
    !kaggle datasets download -d balraj98/indoor-training-set-its-residestandard -p {DATASET_DIR} --unzip
    # NOTE: Tumhe RESIDE-6K ka official link Kaggle se lena hoga. Main example link de raha hu.
    
    print("Dataset download aur extract ho gaya!")
else:
    print("Dataset pehle se majood hai!")

# %% Step 3: Config update karo (Colab ke hisaab se)
config = yaml.safe_load(open('configs/default.yaml'))

# Data Directory Update karo (kyunki data Colab mein '/content/' mein hai)
config['data']['data_dir'] = DATASET_DIR
config['data']['num_workers'] = 2          
config['training']['batch_size'] = 16      
config['data']['image_size'] = 256         
config['training']['epochs'] = 50

with open('configs/colab_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print("Config updated for Colab!")

# %% Step 4: Device check (GPU mil raha hai?)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training chalegi ispe: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% Step 5: DataLoaders Test
from data.dataset import smart_create_dataloaders
config_colab = yaml.safe_load(open('configs/colab_config.yaml'))
train_loader, val_loader, test_loader = smart_create_dataloaders(config_colab)

print("Data load ho gaya asani se!")
batch = next(iter(train_loader))
print(f"Hazy shape: {batch[0].shape}, Clean shape: {batch[1].shape}")

# %% Step 6: MODEL BANANA AUR TRAINING SHURU!
from models.unet import UNet
from models.losses import get_loss_function
from models.metrics import calculate_psnr, calculate_ssim, AverageMeter

model = UNet(
    in_channels=config_colab['model']['in_channels'],
    out_channels=config_colab['model']['out_channels'],
    features=config_colab['model']['features']
).to(device)

criterion = get_loss_function(config_colab)
optimizer = torch.optim.Adam(model.parameters(), lr=config_colab['training']['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, verbose=True
)

epochs = config_colab['training']['epochs']
best_psnr = 0

print(f"\n🚀 Training shuru ho rahi hai {epochs} epochs ke liye...")

for epoch in range(1, epochs + 1):
    model.train()
    loss_meter = AverageMeter()

    for hazy, clean in train_loader:
        hazy, clean = hazy.to(device), clean.to(device)

        optimizer.zero_grad()
        output = model(hazy)
        loss = criterion(output, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item())

    # Validation pass
    model.eval()
    psnr_meter = AverageMeter()
    with torch.no_grad():
        for hazy, clean in val_loader:
            hazy, clean = hazy.to(device), clean.to(device)
            output = model(hazy)
            psnr_meter.update(calculate_psnr(output, clean))

    scheduler.step(loss_meter.avg)

    # Best Model Save
    if psnr_meter.avg > best_psnr:
        best_psnr = psnr_meter.avg
        os.makedirs('experiments', exist_ok=True)
        torch.save(model.state_dict(), 'experiments/best_model_github.pth')

    print(f"Epoch [{epoch}/{epochs}] Loss: {loss_meter.avg:.4f} | Validation PSNR: {psnr_meter.avg:.2f} dB (Best: {best_psnr:.2f} dB)")

# Pura training logic bina kisi manual upload ke chal jayega!
print("\n🎉 Training Mukammal (Complete)!")
