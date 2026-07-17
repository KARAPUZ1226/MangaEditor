# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. АРХИТЕКТУРА УЛЬТРАЛЕГКОЙ U-NET ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return logits

# --- 2. КЛАСС ЗАГРУЗКИ ДАННЫХ (DATASET) ---
class MangaDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.masks_paths = sorted(glob.glob(os.path.join(masks_dir, '*.png')))
        
    def __len__(self):
        return len(self.images_paths)
        
    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        mask_path = self.masks_paths[idx]
        
        # Загрузка изображения (BGR -> RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Загрузка маски (черно-белая)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Нормализация
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Перевод в тензоры PyTorch (Channel First: [C, H, W])
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) # [1, H, W]
        
        return img_tensor, mask_tensor

# --- 3. LOSS ФУНКЦИЯ (BCE + DICE LOSS) ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets, smooth=1.0):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice
        
        return bce_loss + dice_loss

# --- 4. ПРОЦЕСС ОБУЧЕНИЯ ---
def train_model(epochs=10, batch_size=16, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используемое устройство: {device}')
    
    # Пути к датасету
    dataset_dir = 'unet_dataset'
    train_images = os.path.join(dataset_dir, 'train', 'images')
    train_masks = os.path.join(dataset_dir, 'train', 'masks')
    val_images = os.path.join(dataset_dir, 'val', 'images')
    val_masks = os.path.join(dataset_dir, 'val', 'masks')
    
    if not os.path.exists(train_images):
        print(f'Ошибка: Путь {train_images} не существует! Сначала запустите генерацию датасета.')
        return
        
    train_dataset = MangaDataset(train_images, train_masks)
    val_dataset = MangaDataset(val_images, val_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = UNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print('Начало обучения U-Net...')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f'Эпоха {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
    # Сохраняем веса PyTorch
    torch.save(model.state_dict(), 'unet_manga.pth')
    print('Модель успешно сохранена в unet_manga.pth!')
    
    # --- 5. ЭКСПОРТ В ONNX ---
    print('Экспорт модели U-Net в ONNX...')
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    onnx_path = os.path.join('models', 'segmenter.onnx')
    os.makedirs('models', exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['mask'],
        dynamic_axes={'image': {0: 'batch_size'}, 'mask': {0: 'batch_size'}}
    )
    print(f'Модель экспортирована в ONNX: {onnx_path}')

if __name__ == '__main__':
    train_model(epochs=30)
