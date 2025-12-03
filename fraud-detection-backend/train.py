"""
Train Autoencoder Models for Fraud Detection
Trains on: firstName, lastName, photo

Usage:
    python train.py --data cropped/ --output models/ --epochs 300
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VALID_CLASSES = ['firstName', 'lastName', 'photo']
IMAGE_SIZE = 224
LATENT_DIM = 128


class FraudAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2),
        )
        
        self.fc_encode = nn.Linear(512*7*7, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512*7*7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encode(x)
        x = self.fc_decode(x)
        x = x.view(x.size(0), 512, 7, 7)
        return self.decoder(x)
    
    def get_error(self, x):
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon)**2, dim=[1,2,3])


class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.samples = []
        folder = Path(folder)
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.samples.extend(list(folder.glob(ext)))
        print(f"  Loaded {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx]).convert('RGB')
        except:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), 'white')
        return self.transform(img) if self.transform else img


def train_model(data_folder, class_name, output_folder, epochs, batch_size, device):
    output_path = Path(output_folder) / class_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    class_folder = Path(data_folder) / class_name
    
    print(f"\n{'='*60}\nTraining: {class_name}\n{'='*60}")
    
    if not class_folder.exists():
        print(f"Error: {class_folder} not found")
        return None
    
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.1, 0.1, 0.05),
        transforms.ToTensor(),
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = ImageDataset(class_folder, train_tf)
    if len(dataset) < 5:
        print(f"Error: Need at least 5 samples, got {len(dataset)}")
        return None
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = FraudAutoencoder(LATENT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    losses = []
    best_loss = float('inf')
    
    print(f"Training on {len(dataset)} samples...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        
        if (epoch+1) % 30 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path / 'model.pth')
    
    # Calculate threshold
    print("\nCalculating threshold...")
    model.eval()
    val_dataset = ImageDataset(class_folder, val_tf)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    errors = []
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            err = model.get_error(images)
            errors.append(err.item())
    
    errors = np.array(errors)
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))
    threshold = mean_err + 2.5 * std_err
    
    print(f"  Mean: {mean_err:.6f}, Std: {std_err:.6f}, Threshold: {threshold:.6f}")
    
    config = {
        'class_name': class_name,
        'num_samples': len(errors),
        'mean_error': mean_err,
        'std_error': std_err,
        'threshold': threshold,
        'epochs': epochs,
        'latent_dim': LATENT_DIM
    }
    
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses)
    axes[0].set_title(f'{class_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[1].hist(errors, bins=30, alpha=0.7)
    axes[1].axvline(threshold, color='r', linestyle='--', label='Threshold')
    axes[1].set_title(f'{class_name} - Error Distribution')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path / 'training_plot.png')
    plt.close()
    
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', default='cropped')
    parser.add_argument('--output', '-o', default='models')
    parser.add_argument('--epochs', '-e', type=int, default=300)
    parser.add_argument('--batch-size', '-b', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    results = {}
    for cls in VALID_CLASSES:
        cfg = train_model(args.data, cls, args.output, args.epochs, args.batch_size, args.device)
        if cfg:
            results[cls] = cfg
    
    print("\n" + "="*60 + "\nTRAINING COMPLETE\n" + "="*60)
    for cls, cfg in results.items():
        print(f"{cls}: {cfg['num_samples']} samples, threshold={cfg['threshold']:.6f}")


if __name__ == '__main__':
    main()
