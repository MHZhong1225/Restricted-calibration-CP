import os
from types import SimpleNamespace
from typing import Tuple, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import numpy as np

class ImageDatasetWithAttrs(Dataset):
    """Wrapper to make ImageFolder return dummy attributes to match the tabular dataloaders."""
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.num_classes = len(self.dataset.classes)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # a1: 真实分类标签
        # a2: 占位
        # a3: 占位
        return x, y, y, 0, 0

def build_dataloaders_bach(cfg: SimpleNamespace) -> Tuple[Any, Any, Any, Any]:
    data_dir = getattr(cfg, "image_data_dir", "../../BrCPT/datasets/bach")
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
        
    # Standard ResNet normalization and resizing
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageDatasetWithAttrs(train_dir, transform=transform_train)
    
    # If val/test don't exist, we might need to split train
    if os.path.exists(val_dir) and os.path.exists(test_dir):
        cal_dataset = ImageDatasetWithAttrs(val_dir, transform=transform_test)
        test_dataset = ImageDatasetWithAttrs(test_dir, transform=transform_test)
    else:
        # Fallback split if val/test folders aren't there
        from torch.utils.data import random_split
        total_len = len(train_dataset)
        train_len = int(0.6 * total_len)
        cal_len = int(0.2 * total_len)
        test_len = total_len - train_len - cal_len
        
        # We need to apply test transforms to val/test splits
        # This is a bit hacky but works for the fallback case
        full_dataset = ImageDatasetWithAttrs(train_dir, transform=None)
        
        generator = torch.Generator().manual_seed(cfg.seed)
        train_ds, cal_ds, test_ds = random_split(full_dataset, [train_len, cal_len, test_len], generator=generator)
        
        # Apply transforms
        class SplitDataset(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            def __len__(self): return len(self.subset)
            def __getitem__(self, idx):
                x, y, a1, a2, a3 = self.subset[idx]
                if self.transform:
                    x = self.transform(x)
                return x, y, a1, a2, a3
                
        train_dataset = SplitDataset(train_ds, transform_train)
        cal_dataset = SplitDataset(cal_ds, transform_test)
        test_dataset = SplitDataset(test_ds, transform_test)
    
    batch_size = getattr(cfg, "batch_size", 32)
    
    # Use multiple workers for faster image loading
    num_workers = 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    cal_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    meta = SimpleNamespace(
        dataset_mode="bach",
        path=data_dir,
        n_train=len(train_dataset),
        n_cal=len(cal_dataset),
        n_test=len(test_dataset),
        num_classes=train_dataset.dataset.num_classes if hasattr(train_dataset, 'dataset') else 4,
        feature_dim=None, # Will be set based on backbone
        is_image=True
    )
    
    return train_loader, cal_loader, test_loader, meta
