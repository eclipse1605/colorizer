import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import gc

from model import ECCVColorizer, SIGGRAPHColorizer
from data import get_dataloaders, prepare_dataset_from_download
from utils import (
    lab_tensors_to_rgb, create_directories, calculate_metrics,
    unpad_tensor, visualize_colorization, pad_tensor, tensor_to_numpy, save_result, visualize_result, 
    numpy_to_tensor, postprocess_output
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Colorization Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='eccv', choices=['eccv', 'siggraph'], 
                        help='Model type: eccv or siggraph')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')
    
    return parser.parse_args()


def set_gpu_limits(power_limit=0, gpu_freq=0):
    """Set GPU power and frequency limits to prevent crashes"""
    try:
                         
        if not torch.cuda.is_available():
            return
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return
        
                                      
        if power_limit > 0:
            for i in range(device_count):
                                                                  
                result = subprocess.run(
                    ['nvidia-smi', f'-i', f'{i}', '--query-gpu=power.limit', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True
                )
                max_power = float(result.stdout.strip())
                safe_power = min(power_limit, max_power * 0.8)                                 
                
                                   
                subprocess.run(
                    ['nvidia-smi', f'-i', f'{i}', f'--power-limit={int(safe_power)}'],
                    capture_output=True
                )
                print(f"GPU {i} power limit set to {safe_power}W")
        
                                              
        if gpu_freq > 0:
            for i in range(device_count):
                subprocess.run(
                    ['nvidia-smi', f'-i', f'{i}', f'--lock-gpu-clocks={gpu_freq},{gpu_freq}'],
                    capture_output=True
                )
                print(f"GPU {i} clock locked to {gpu_freq}MHz")
    except Exception as e:
        print(f"Warning: Failed to set GPU limits: {e}")


def clear_gpu_memory():
    """Aggressive GPU memory clearing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()


def train_model():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    samples_dir = os.path.join(args.output_dir, 'samples')
    logs_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create dataloader
    train_loader, val_loader = get_dataloaders(
        args.data_dir, args.batch_size, args.img_size,
        num_workers=4, split_ratio=0.9
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    if args.model_type == 'eccv':
        model = ECCVColorizer().to(device)
        print("Using ECCV model")
    else:
        model = SIGGRAPHColorizer().to(device)
        print("Using SIGGRAPH model")
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize tensorboard
    writer = SummaryWriter(logs_dir)
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        # Progress bar
        train_pbar = tqdm(train_loader)
        
        for batch in train_pbar:
            # Get data
            L = batch['L'].to(device)
            ab = batch['ab'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            ab_pred = model(L)
            
            # Calculate loss
            loss = criterion(ab_pred, ab)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            train_pbar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")
            
            # Log to tensorboard
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            global_step += 1
        
        # Average train loss
        train_loss /= len(train_loader)
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(val_pbar):
                # Get data
                L = batch['L'].to(device)
                ab = batch['ab'].to(device)
                
                # Forward pass
                ab_pred = model(L)
                
                # Calculate loss
                loss = criterion(ab_pred, ab)
                val_loss += loss.item()
                
                # Save some samples for visualization
                if batch_idx < 5:  # Save first 5 samples
                    for i in range(min(3, L.size(0))):  # Sample 3 images
                        val_samples.append({
                            'L': L[i],
                            'ab_pred': ab_pred[i],
                            'ab_true': ab[i]
                        })
        
        # Average validation loss
        val_loss /= len(val_loader)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(checkpoints_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth'))
        
        # Visualize some validation samples
        if val_samples:
            fig, axes = plt.subplots(len(val_samples), 3, figsize=(12, 4 * len(val_samples)))
            
            for i, sample in enumerate(val_samples):
                L = sample['L'].cpu()
                ab_pred = sample['ab_pred'].cpu()
                ab_true = sample['ab_true'].cpu()
                
                # Denormalize L
                L_np = tensor_to_numpy(L.squeeze(0))
                L_np = (L_np + 1.0) * 50.0
                
                # Denormalize ab - fix transpose to work with PyTorch tensors
                # For PyTorch tensors, we need to do permute instead of transpose with 3 dimensions
                # Or transpose dimensions one at a time, then convert to numpy
                ab_pred_np = tensor_to_numpy(ab_pred.permute(1, 2, 0)) * 110.0
                ab_true_np = tensor_to_numpy(ab_true.permute(1, 2, 0)) * 110.0
                
                # Convert to RGB
                from utils import lab_to_rgb
                pred_rgb = lab_to_rgb(L_np, ab_pred_np)
                true_rgb = lab_to_rgb(L_np, ab_true_np)
                
                # Display grayscale, prediction, and ground truth
                if len(val_samples) == 1:
                    row_axes = axes
                else:
                    row_axes = axes[i]
                
                row_axes[0].imshow(L_np, cmap='gray')
                row_axes[0].set_title('Grayscale')
                row_axes[0].axis('off')
                
                row_axes[1].imshow(pred_rgb)
                row_axes[1].set_title('Prediction')
                row_axes[1].axis('off')
                
                row_axes[2].imshow(true_rgb)
                row_axes[2].set_title('Ground Truth')
                row_axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'epoch_{epoch+1}.png'))
            plt.close()
            
            # Add figure to tensorboard
            writer.add_figure('Validation Samples', fig, epoch)
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
    }, os.path.join(checkpoints_dir, 'final_model.pth'))
    
    writer.close()
    print("Training complete!")


if __name__ == '__main__':
    train_model() 