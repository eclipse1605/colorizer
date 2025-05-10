import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from utils import rgb_to_lab, resize_img

class ColorizationDataset(Dataset):
    """Dataset for image colorization"""
    def __init__(self, image_paths, img_size=256, split='train'):
        """
        Args:
            image_paths (list): List of paths to RGB images
            img_size (int): Size to resize images to
            split (str): 'train' or 'val' or 'test' - determines augmentation
        """
        self.image_paths = image_paths
        self.img_size = img_size
        self.split = split
        
        # Data augmentation for training only
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.transform = transforms.Resize((img_size, img_size))
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.split == 'train':
                img = self.transform(img)
            else:
                img = self.transform(img)
                
            # Convert to numpy for Lab conversion
            img_np = np.array(img)
            
            # Get L and ab channels
            L, ab = rgb_to_lab(img_np)
            
            # Normalize L to range [-1, 1]
            L = L / 50.0 - 1.0
            
            # Normalize ab to range [-1, 1]
            ab = ab / 110.0
            
            # Convert to tensors
            L_tensor = torch.from_numpy(L).unsqueeze(0).float()
            ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)).float()
            
            return {'L': L_tensor, 'ab': ab_tensor, 'path': img_path}
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a placeholder in case of error
            L_tensor = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)
            ab_tensor = torch.zeros((2, self.img_size, self.img_size), dtype=torch.float32)
            return {'L': L_tensor, 'ab': ab_tensor, 'path': img_path}

def create_dataloaders(image_dir, batch_size=16, img_size=256, num_workers=4, split_ratio=0.9):
    """Create train and validation dataloaders from a directory of images
    
    Args:
        image_dir (str): Directory containing images
        batch_size (int): Batch size
        img_size (int): Size to resize images to
        num_workers (int): Number of workers for DataLoader
        split_ratio (float): Ratio of train to validation split
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get all image paths
    extensions = ('.jpg', '.jpeg', '.png')
    image_paths = [
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.lower().endswith(extensions)
    ]
    
    # Split into train and validation
    np.random.shuffle(image_paths)
    split_idx = int(len(image_paths) * split_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Create datasets
    train_dataset = ColorizationDataset(train_paths, img_size, split='train')
    val_dataset = ColorizationDataset(val_paths, img_size, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 