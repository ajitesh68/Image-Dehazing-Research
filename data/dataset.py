import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DehazingDataset(Dataset):

    def __init__(self, hazy_dir, clean_dir, image_size=128, transform=None):
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir
        self.image_size = image_size
        self.transform = transform
        self.hazy_images = sorted(os.listdir(hazy_dir))
        self.clean_images = sorted(os.listdir(clean_dir))
        assert len(self.hazy_images) == len(self.clean_images), f'Mismatch! {len(self.hazy_images)} hazy images vs {len(self.clean_images)} clean images'
        self.base_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, index):
        hazy_path = os.path.join(self.hazy_dir, self.hazy_images[index])
        clean_path = os.path.join(self.clean_dir, self.clean_images[index])
        hazy_image = Image.open(hazy_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')
        hazy_tensor = self.base_transform(hazy_image)
        clean_tensor = self.base_transform(clean_image)
        if self.transform is not None:
            seed = random.randint(0, 2 ** 32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)
            hazy_tensor = self.transform(hazy_tensor)
            random.seed(seed)
            torch.manual_seed(seed)
            clean_tensor = self.transform(clean_tensor)
        return (hazy_tensor, clean_tensor)

def create_dataloaders(config):
    data_dir = config['data']['data_dir']
    image_size = config['data']['image_size']
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']
    hazy_dir = os.path.join(data_dir, 'hazy')
    clean_dir = os.path.join(data_dir, 'GT')
    full_dataset = DehazingDataset(hazy_dir=hazy_dir, clean_dir=clean_dir, image_size=image_size)
    total_size = len(full_dataset)
    train_size = int(config['data']['train_split'] * total_size)
    val_size = int(config['data']['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    print(f'Dataset splits — Train: {train_size}, Val: {val_size}, Test: {test_size}')
    return (train_loader, val_loader, test_loader)

def create_dataloaders_presplit(config):
    data_dir = config['data']['data_dir']
    image_size = config['data']['image_size']
    batch_size = config['training']['batch_size']
    num_workers = config['data']['num_workers']
    train_hazy = os.path.join(data_dir, 'train', 'hazy')
    train_clean = os.path.join(data_dir, 'train', 'GT')
    test_hazy = os.path.join(data_dir, 'test', 'hazy')
    test_clean = os.path.join(data_dir, 'test', 'GT')
    for path, name in [(train_hazy, 'train/hazy'), (train_clean, 'train/GT'), (test_hazy, 'test/hazy'), (test_clean, 'test/GT')]:
        assert os.path.exists(path), f"Folder nahi mila: {path}\nRESIDE-6K mein '{name}' folder hona chahiye!"
    from data.augmentations import get_train_augmentations, get_val_augmentations
    train_aug = get_train_augmentations()
    full_train = DehazingDataset(hazy_dir=train_hazy, clean_dir=train_clean, image_size=image_size, transform=train_aug)
    total_train = len(full_train)
    val_size = int(0.15 * total_train)
    train_size = total_train - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_train, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    test_dataset = DehazingDataset(hazy_dir=test_hazy, clean_dir=test_clean, image_size=image_size, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    print(f'RESIDE-6K Dataset loaded!')
    print(f'  Train: {train_size} images (augmented)')
    print(f'  Val:   {val_size} images (from training split)')
    print(f'  Test:  {len(test_dataset)} images (untouched)')
    return (train_loader, val_loader, test_loader)

def smart_create_dataloaders(config):
    data_dir = config['data']['data_dir']
    if os.path.exists(os.path.join(data_dir, 'train')):
        print('Pre-split dataset detected (RESIDE-6K style)')
        return create_dataloaders_presplit(config)
    else:
        print('Single-folder dataset detected (custom style)')
        return create_dataloaders(config)